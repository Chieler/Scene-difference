"""Object-level change detection on the SceneDiff eval frames.

1. VGGT reconstructs the eval frames of both videos jointly (shared frame, no alignment step).
2. Per-pixel change maps for each frame: DINOv2 patch dissimilarity to the same 3D spot in the other video, plus a
   free-space term (the other video sees past this point, so the surface is gone there).
3. Change regions become SAM2 masks (box + peak-point prompts).
4. Masks are lifted to 3D and merged across frames of one video into objects, then projected into the frames where
   no mask fired and re-segmented there, so every frame that sees the object gets a mask.
5. A removed object and an added object that look alike become one "moved" object.

    uv run objdiff.py [scene ...] -o out/pred_obj
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import bench
import recon
from change2d import UPSCALE, dino, nearest_frames

SAM = "facebook/sam2.1-hiera-base-plus"
P = dict(
    occ_tol=0.05,  # other view's depth this much (relative) in front: occluded, no appearance evidence
    w_geo=1.0,  # weight of the free-space term
    geo_cap=0.3,  # free-space term saturates at this relative depth gap
    thr=0.45,  # change-map threshold for proposals
    min_patches=4,  # smallest proposal, in DINO patches (2x upscaled 14 px at 518 long side)
    max_area=0.25,  # masks covering more of the frame are walls/floor, not objects
    min_score=0.35,  # mean change inside the SAM mask
    voxel=0.03,  # association voxel, as a fraction of median scene depth
    link=0.25,  # voxel overlap (of the smaller set) to merge two masks into one object
    moved_sim=0.75,  # DINO cosine between a removed and an added object to call it moved
)


def recon_scene(scene, frames, cache):
    if cache.exists():
        z = np.load(cache)
        return [recon.Views(*(z[f"{k}{v}"] for k in ("images", "depth", "conf", "K", "E"))) for v in (1, 2)]
    size = recon.frame_size(frames[1][0])
    imgs = recon.load_frames(frames[1] + frames[2], size)
    depth, conf, K, E = recon.run_vggt(imgs)
    v = recon.Views(imgs, depth, conf, K, E)
    n = len(frames[1])
    va, vb = v.part(0, n), v.part(n, len(imgs))
    np.savez_compressed(cache, **{f"{k}{i}": getattr(x, k) for i, x in ((1, va), (2, vb))
                                  for k in ("images", "depth", "conf", "K", "E")})
    return va, vb


def change_maps(src, dst, fs, fd):
    """(S, h, w) change score per patch of each src frame; NaN where no dst frame sees the spot."""
    app, geo = change_terms(src, dst, fs, fd)
    return combine(app, geo)


def combine(app, geo):
    return app + P["w_geo"] * np.clip(np.nan_to_num(geo), 0, P["geo_cap"]) / P["geo_cap"] * 0.5


def change_terms(src, dst, fs, fd):
    """Appearance term (1 - best DINO similarity, NaN where unseen) and free-space term (median relative depth gap
    in the other video's views: > 0 means they see past this point) per patch of each src frame."""
    S, H, W = src.depth.shape
    _, h, w, _ = fs.shape
    yi = ((np.arange(h) + 0.5) * H / h).astype(int)
    xi = ((np.arange(w) + 0.5) * W / w).astype(int)
    vv, uu = np.meshgrid(yi, xi, indexing="ij")
    near = nearest_frames(src, dst, k=min(8, len(dst.depth)))
    app = np.full((S, h, w), np.nan, np.float32)
    geo = np.full((S, h, w), np.nan, np.float32)
    for i in range(S):
        d = src.depth[i][vv, uu].reshape(-1)
        K, R, t = src.K[i], src.E[i, :, :3], src.E[i, :, 3]
        cam = np.stack([(uu.reshape(-1) - K[0, 2]) / K[0, 0] * d, (vv.reshape(-1) - K[1, 2]) / K[1, 1] * d, d], -1)
        X = (cam - t) @ R
        f = fs[i].reshape(-1, fs.shape[-1]).astype(np.float32)
        best = np.full(len(X), np.nan, np.float32)
        gaps = []
        for j in near[i]:
            c = X @ dst.E[j, :, :3].T + dst.E[j, :, 3]
            z = c[:, 2]
            uv = c @ dst.K[j].T
            u, v = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
            inb = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            gap = np.full(len(X), np.nan, np.float32)
            dj = dst.depth[j][v[inb].astype(int), u[inb].astype(int)]
            gap[inb] = (dj - z[inb]) / z[inb]
            gaps.append(gap)
            ok = inb.copy()
            ok[inb] &= gap[inb] > -P["occ_tol"]
            if not ok.any():
                continue
            pa, pb = (v[ok] * h / H).astype(int), (u[ok] * w / W).astype(int)
            sim = np.full(ok.sum(), -1.0, np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    g = fd[j][np.clip(pa + dy, 0, h - 1), np.clip(pb + dx, 0, w - 1)].astype(np.float32)
                    sim = np.maximum(sim, np.einsum("nc,nc->n", g, f[ok]))
            best[ok] = np.fmax(best[ok], sim)
        with np.errstate(all="ignore"):
            geo[i] = np.nanmedian(np.stack(gaps), 0).reshape(h, w)
        app[i] = (1 - best).reshape(h, w)
    return app, geo


class Segmenter:
    def __init__(self):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch, self.dev = torch, "mps" if torch.backends.mps.is_available() else "cpu"
        self.m = Sam2Model.from_pretrained(SAM).to(self.dev).eval()
        self.p = Sam2Processor.from_pretrained(SAM)
        self.calls = 0

    def __call__(self, img, boxes, points=None):
        """boxes [(x1,y1,x2,y2)], points [(x,y)] or None -> (N, H, W) bool masks, (N,) predicted IoU."""
        if not boxes:
            return np.zeros((0,) + img.shape[:2], bool), np.zeros(0)
        kw = dict(input_boxes=[[list(map(float, b)) for b in boxes]])
        if points is not None:
            kw.update(input_points=[[[list(map(float, p))] for p in points]], input_labels=[[[1] for _ in points]])
        self.calls += 1
        inp = self.p(images=img, return_tensors="pt", **kw).to(self.dev)
        with self.torch.no_grad():
            emb = self.m.get_image_embeddings(inp["pixel_values"])
            o = self.m(image_embeddings=emb, input_boxes=inp["input_boxes"], multimask_output=False,
                       **({"input_points": inp["input_points"], "input_labels": inp["input_labels"]} if points else {}))
        ms = self.p.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][:, 0].numpy()
        return ms, o.iou_scores.flatten().cpu().float().numpy()


def lift(mask, v, i, shape, stride=4):
    """World points under a full-res mask, via VGGT depth (sampled at the VGGT resolution)."""
    H, W = v.depth.shape[1:]
    m = ndimage.zoom(mask.astype(np.uint8), (H / shape[0], W / shape[1]), order=0).astype(bool)
    ys, xs = np.nonzero(m)
    ys, xs = ys[::stride], xs[::stride]
    d = v.depth[i][ys, xs]
    K, R, t = v.K[i], v.E[i, :, :3], v.E[i, :, 3]
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], -1)
    return (cam - t) @ R


def project(X, v, i, shape):
    """Project world points into frame i; keep those not hidden behind the frame's depth. Full-res pixel coords."""
    H, W = v.depth.shape[1:]
    c = X @ v.E[i, :, :3].T + v.E[i, :, 3]
    z = c[:, 2]
    uv = c @ v.K[i].T
    u, w_ = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
    ok = (z > 1e-3) & (u >= 0) & (u < W) & (w_ >= 0) & (w_ < H)
    ok[ok] &= v.depth[i][w_[ok].astype(int), u[ok].astype(int)] > z[ok] * 0.9
    return np.stack([u[ok] * shape[1] / W, w_[ok] * shape[0] / H], -1)


def proposals(cmap, shape):
    """Connected high-change regions of one (h, w) map -> boxes and peak points in full-res pixels, region score."""
    h, w = cmap.shape
    m = np.nan_to_num(cmap) > P["thr"]
    lab, n = ndimage.label(m)
    sy, sx = shape[0] / h, shape[1] / w
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        r = lab[sl] == k
        if r.sum() < P["min_patches"]:
            continue
        vals = np.where(r, cmap[sl], -np.inf)
        py, px = np.unravel_index(np.argmax(vals), r.shape)
        box = (sl[1].start * sx, sl[0].start * sy, sl[1].stop * sx, sl[0].stop * sy)
        out.append((box, ((px + sl[1].start + 0.5) * sx, (py + sl[0].start + 0.5) * sy)))
    return out


def mask_score(mask, cmap):
    m = ndimage.zoom(mask.astype(np.uint8), (cmap.shape[0] / mask.shape[0], cmap.shape[1] / mask.shape[1]), order=0)
    vals = cmap[m.astype(bool)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def keys(X, vox):
    return set(map(tuple, np.floor(X / vox).astype(np.int64)))


def detect_video(imgs, v, cmaps, feats, seg, vox):
    """Masks in each frame of one video -> objects: {'frames': {i: mask}, 'scores': {i: s}, 'X': points}."""
    shape = imgs[0].shape[:2]
    dets = []
    for i, img in enumerate(imgs):
        pr = proposals(cmaps[i], shape)
        if not pr:
            continue
        ms, iou = seg(img, [b for b, _ in pr], [p for _, p in pr])
        for m in ms:
            a = m.mean()
            if a < 1e-4 or a > P["max_area"]:
                continue
            s = mask_score(m, cmaps[i])
            if s < P["min_score"]:
                continue
            X = lift(m, v, i, shape)
            if len(X) < 5:
                continue
            dets.append(dict(i=i, mask=m, score=s, X=X, K=keys(X, vox)))
    # union-find over masks whose 3D voxels overlap
    par = list(range(len(dets)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for a in range(len(dets)):
        for b in range(a + 1, len(dets)):
            ka, kb = dets[a]["K"], dets[b]["K"]
            if len(ka & kb) >= P["link"] * min(len(ka), len(kb)):
                par[find(a)] = find(b)
    groups = {}
    for a in range(len(dets)):
        groups.setdefault(find(a), []).append(dets[a])
    objs = []
    for g in groups.values():
        frames, scores = {}, {}
        for d in g:
            frames[d["i"]] = frames.get(d["i"], np.zeros(shape, bool)) | d["mask"]
            scores[d["i"]] = max(scores.get(d["i"], 0), d["score"])
        objs.append(dict(frames=frames, scores=scores, X=np.concatenate([d["X"] for d in g])))
    # fill frames that see the object but where no mask fired: one SAM pass per frame for all objects
    base = {id(o): float(np.mean(list(o["scores"].values()))) for o in objs}
    for i, img in enumerate(imgs):
        need, boxes, pts = [], [], []
        for o in objs:
            if i in o["frames"]:
                continue
            uv = project(o["X"], v, i, shape)
            if len(uv) < 10:
                continue
            lo, hi = np.percentile(uv, 2, 0), np.percentile(uv, 98, 0)
            need.append(o); boxes.append((*lo, *hi)); pts.append(tuple(np.median(uv, 0)))
        if not need:
            continue
        ms, _ = seg(img, boxes, pts)
        for o, m in zip(need, ms):
            if m.any() and m.mean() <= P["max_area"]:
                o["frames"][i], o["scores"][i] = m, base[id(o)] * 0.8
    for o in objs:
        o["score"] = float(np.mean(list(o["scores"].values())))
        o["feat"] = embed(o, imgs, feats)
    return objs


def embed(o, imgs, feats):
    """Mean DINO patch feature under the object's masks."""
    acc = []
    for i, m in o["frames"].items():
        f = feats[i]
        mm = ndimage.zoom(m.astype(np.uint8), (f.shape[0] / m.shape[0], f.shape[1] / m.shape[1]), order=0).astype(bool)
        if mm.any():
            acc.append(f[mm].astype(np.float32).mean(0))
    e = np.mean(acc, 0) if acc else np.zeros(feats.shape[-1], np.float32)
    return e / (np.linalg.norm(e) + 1e-8)


def pair_moved(o1, o2):
    """Greedy: highest-similarity removed/added pairs above moved_sim become one object."""
    if not o1 or not o2:
        return [(a, None) for a in o1] + [(None, b) for b in o2]
    S = np.array([[a["feat"] @ b["feat"] for b in o2] for a in o1])
    used1, used2, out = set(), set(), []
    for k in np.argsort(-S, axis=None):
        a, b = np.unravel_index(k, S.shape)
        if S[a, b] < P["moved_sim"]:
            break
        if a in used1 or b in used2:
            continue
        used1.add(a); used2.add(b)
        out.append((o1[a], o2[b]))
    out += [(o1[a], None) for a in range(len(o1)) if a not in used1]
    out += [(None, o2[b]) for b in range(len(o2)) if b not in used2]
    return out


def run_scene(scene, out_dir, seg, maps_only=False):
    work = Path("out/obj") / scene
    work.mkdir(parents=True, exist_ok=True)
    frames = {v: bench.extract(scene, v, Path("out/frames") / scene) for v in (1, 2)}
    imgs = {v: bench.load_frames(frames[v]) for v in (1, 2)}
    va, vb = recon_scene(scene, frames, work / "recon.npz")
    t = time.time()
    fa, fb = dino(va.images), dino(vb.images)
    (a1, g1), (a2, g2) = change_terms(va, vb, fa, fb), change_terms(vb, va, fb, fa)
    ca, cb = combine(a1, g1), combine(a2, g2)
    np.savez_compressed(work / "cmaps.npz", c1=ca, c2=cb, a1=a1, g1=g1, a2=a2, g2=g2)
    if maps_only:
        return 0, 0, 0
    print(f"  change maps {time.time() - t:.0f}s", flush=True)
    t = time.time()
    vox = P["voxel"] * float(np.median(np.concatenate([va.depth.ravel(), vb.depth.ravel()])))
    o1 = detect_video(imgs[1], va, ca, fa, seg, vox)
    o2 = detect_video(imgs[2], vb, cb, fb, seg, vox)
    print(f"  segmentation {time.time() - t:.0f}s, {seg.calls} SAM calls", flush=True)
    objects = []
    for a, b in pair_moved(o1, o2):
        o, fs = {}, {}
        for vk, x in (("video_1", a), ("video_2", b)):
            if x is None:
                continue
            o[vk] = x["frames"]
            fs.update({(vk, i): s for i, s in x["scores"].items()})
        o["score"] = float(np.mean([x["score"] for x in (a, b) if x is not None]))
        o["frame_scores"] = fs
        objects.append(o)
    H, W = imgs[1][0].shape[:2]
    bench.save_pred(Path(out_dir) / scene / "object_masks.pkl", objects, H, W)
    return len(o1), len(o2), len(objects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("-o", "--out", default="out/pred_obj")
    ap.add_argument("--maps-only", action="store_true", help="stop after reconstruction and change maps")
    ap.add_argument("--set", action="append", default=[], help="override a parameter, e.g. thr=0.5")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = type(P[k])(v)
    seg = None if a.maps_only else Segmenter()
    todo = a.scenes or [s for s in bench.scenes() if (bench.DATA / "data" / s).exists()]
    log = []
    for s in todo:
        t = time.time()
        n = run_scene(s, a.out, seg, a.maps_only)
        log.append(dict(scene=s, v1=n[0], v2=n[1], objects=n[2], seconds=round(time.time() - t, 1)))
        print(log[-1], flush=True)
    json.dump(dict(params=P, scenes=log), open(Path(a.out) / "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
