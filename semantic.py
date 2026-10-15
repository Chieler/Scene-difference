# semantic.py
"""Object changes proposed by a VLM, checked by geometry.

A vision-language model (vlm.py: OpenAI API or the codex CLI) sees both videos as labeled contact sheets and lists the objects it thinks were
added, removed or moved, with a name, a sentence and a box in each frame where it shows. SAM2 turns the boxes into
masks and VGGT depth lifts the masks into the shared 3D frame.

The model is good at noticing and naming, poor at telling a real change from a new viewpoint. Geometry is the
other way round, so every claim is tested in 3D against the other video:
- seen:    the other video looked at this spot (in view, not hidden behind something nearer);
- gone:    where it looked, it saw *through* the spot (free space where the object stood);
- differs: where it looked, the DINO patch there does not look like the object.
A claim the other video never looked at is kept as "unverified", one the geometry contradicts is "rejected", and a
"moved" object whose two positions coincide is rejected too.
"""
import io, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import change2d, recon

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
SAM = "facebook/sam2.1-hiera-base-plus"
TILE, COLS = 480, 3  # contact sheet: tiles fit a TILE x TILE box, COLS per row, 2 rows
OCC_TOL = 0.08  # relative depth slack: nearer than (1 - tol) occludes, farther than (1 + tol) is free space
MIN_SEEN = 0.25  # fraction of the object's points the other video must see to test the claim at all
MIN_SUPPORT = 0.5  # evidence above the background ring needed to keep a tested claim
DIFF_SCALE = 0.3  # an appearance change this much above the ring's counts like full free-space evidence
MIN_AREA = 0.005  # mean mask area as a fraction of the frame; smaller objects span too few DINO patches to test
MIN_MOVE = 0.15  # m; a "moved" object displaced less than this did not move

PROMPT = """These contact sheets show two phone videos of the same room, recorded at different times.
Video A is before, video B is after. Each tile is one frame; its label gives the video and frame number.

List the physical objects that changed between A and B:
- added: visible in B only (it was not there in A);
- removed: visible in A only;
- moved: the same object at a clearly different place in A and in B.
Ignore differences that only come from camera viewpoint, occlusion, lighting, reflections or people.
Something you only see in one video because the other never pointed there is NOT a change.

For each change give a short name, a one-sentence description that says where it is relative to nearby furniture,
an honest confidence from 0 to 1, and a tight box around the object in EVERY tile where any part of it is visible
(for moved objects, in both videos). Boxes are [x_min, y_min, x_max, y_max] from 0 to 1000 relative to the tile.
Return no objects if nothing changed."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {"objects": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["name", "change", "description", "confidence", "boxes"], "properties": {
        "name": {"type": "string"}, "change": {"type": "string", "enum": ["added", "removed", "moved"]},
        "description": {"type": "string"}, "confidence": {"type": "number"},
        "boxes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                             "required": ["video", "frame", "box"], "properties": {
            "video": {"type": "string", "enum": ["A", "B"]}, "frame": {"type": "integer"},
            "box": {"type": "array", "items": {"type": "integer"}}}}}}}}}}


def sheets(paths, video):
    """Labeled contact sheets of one video's frames, 2 rows of COLS tiles each."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    s = TILE / max(ims[0].size)
    tw, th = round(ims[0].width * s), round(ims[0].height * s)
    out = []
    for k in range(0, len(ims), 2 * COLS):
        sheet = Image.new("RGB", (COLS * tw, 2 * (th + 22)), "#202124")
        d = ImageDraw.Draw(sheet)
        for j, im in enumerate(ims[k:k + 2 * COLS]):
            x, y = j % COLS * tw, j // COLS * (th + 22)
            sheet.paste(im.resize((tw, th), Image.BICUBIC), (x, y + 22))
            d.text((x + 4, y + 4), f"video {video} frame {k + j}", fill="white")
        out.append(sheet)
    return out


def ask_vlm(pa, pb, cache):
    """The model's change claims (parsed JSON), cached by frame set and model."""
    key = f"{MODEL}|{[str(p) for p in pa]}|{[str(p) for p in pb]}"
    try:
        z = json.loads(cache.read_text())
        if z["key"] == key:
            return z["objects"]
    except Exception:
        pass
    import vlm  # OpenAI API, or the codex CLI
    objects = vlm.ask(PROMPT, sheets(pa, "A") + sheets(pb, "B"), SCHEMA, "changes")["objects"]
    cache.write_text(json.dumps({"key": key, "objects": objects}, indent=1))
    return objects


def _jpeg(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=90); return b.getvalue()


def segment(objects, frames):
    """{(object, video, frame): bool mask at full frame resolution}, one SAM2 pass per frame."""
    import torch
    from transformers import Sam2Model, Sam2Processor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc, model = Sam2Processor.from_pretrained(SAM), Sam2Model.from_pretrained(SAM).to(dev).eval()
    jobs = {}
    for k, o in enumerate(objects):
        for b in o["boxes"]:
            if 0 <= b["frame"] < len(frames[b["video"]]) and len(b["box"]) == 4:
                jobs.setdefault((b["video"], b["frame"]), []).append((k, b["box"]))
    out = {}
    for (v, f), items in jobs.items():
        im = Image.open(frames[v][f]).convert("RGB")
        boxes = [[sorted((x0, x1))[0] * im.width / 1000, sorted((y0, y1))[0] * im.height / 1000,
                  sorted((x0, x1))[1] * im.width / 1000, sorted((y0, y1))[1] * im.height / 1000]
                 for _, (x0, y0, x1, y1) in items]
        x = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
        with torch.no_grad():
            r = model(**x, multimask_output=False)
        m = proc.post_process_masks(r.pred_masks.cpu(), x["original_sizes"].cpu())[0][:, 0].numpy()
        for (k, _), mk in zip(items, m):
            if mk.any():
                out[(k, v, f)] = mk
    return out


def lift(mask, v, i, feats):
    """World points under a mask in frame i with the DINO feature at each, and the same for a ring of background
    around the mask. Drops the mask's edge pixels and depth outliers (bleed onto the background behind it)."""
    H, W = v.depth.shape[1:]
    m = np.asarray(Image.fromarray(mask).resize((W, H), Image.NEAREST))
    ring = ndimage.binary_dilation(m, iterations=max(3, int(0.3 * np.sqrt(m.sum())))) & ~ndimage.binary_dilation(m, iterations=2)
    m = ndimage.binary_erosion(m, iterations=1) if m.sum() > 40 else m
    X = recon.unproject(v.depth[i], v.K[i], v.E[i])
    h, w = feats.shape[1:3]
    pick = lambda ys, xs: (X[ys, xs], feats[i][ys * h // H, xs * w // W].astype(np.float32))
    ys, xs = np.nonzero(m)
    d = v.depth[i][ys, xs]
    if len(d) < 5:
        return np.zeros((0, 3)), np.zeros((0, feats.shape[-1])), *pick(*np.nonzero(ring))
    ok = np.abs(d - np.median(d)) < 3 * (np.median(np.abs(d - np.median(d))) + 1e-6)
    return (*pick(ys[ok], xs[ok]), *pick(*np.nonzero(ring)))


def evidence(X, fx, other, fo):
    """Test object points X (features fx) against every frame of the other video. Returns the fraction of points
    it sees, and among its observations the fraction that look through the spot and the mean appearance change."""
    H, W = other.depth.shape[1:]
    h, w = fo.shape[1:3]
    seen = np.zeros(len(X), bool)
    best = np.full(len(X), -1.0, np.float32)
    n_obs = n_gone = 0
    for j in range(len(other.E)):
        c = X @ other.E[j, :, :3].T + other.E[j, :, 3]
        z = c[:, 2]
        uv = c @ other.K[j].T
        u, v = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
        ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        dj = other.depth[j][v[ok].astype(int), u[ok].astype(int)]
        vis = dj > z[ok] * (1 - OCC_TOL)  # nothing nearer in the way
        if not vis.any():
            continue
        idx = np.flatnonzero(ok)[vis]
        seen[idx] = True
        n_obs += len(idx)
        n_gone += int((dj[vis] > z[idx] * (1 + OCC_TOL)).sum())
        g = fo[j][(v[idx] * h / H).astype(int), (u[idx] * w / W).astype(int)].astype(np.float32)
        best[idx] = np.maximum(best[idx], np.einsum("nc,nc->n", g, fx[idx]))
    return {"seen": float(seen.mean()) if len(X) else 0.0,
            "gone": n_gone / n_obs if n_obs else 0.0,
            "differs": float(1 - best[seen].mean()) if seen.any() else 0.0}


def support(e, ctx):
    """How much more the object changed than the background ring around it, which carries the same registration
    noise: either free space or appearance is enough."""
    return max(e["gone"] - ctx["gone"], (e["differs"] - ctx["differs"]) / DIFF_SCALE)


def judge(kind, ev, ctx, shift, area):
    """'confirmed', 'unverified' (the other video never looked) or 'rejected', with the reason."""
    if kind == "moved":
        if shift is None:
            return "rejected", "only found in one video"
        if shift < MIN_MOVE:
            return "rejected", f"same place in both videos ({shift:.2f} m apart)"
        return "confirmed", f"moved {shift:.2f} m"
    e = ev[0]
    if e["seen"] < MIN_SEEN:
        return "unverified", "the other video never looked at this spot"
    if area < MIN_AREA:
        return "unverified", "too small to check in 3D"
    s = support(e, ctx[0])
    if s < MIN_SUPPORT:
        return "rejected", "the other video sees the same thing there"
    return "confirmed", ("the other video sees through this spot" if e["gone"] - ctx[0]["gone"] >= s
                         else "the other video sees something else there")


def detect(A, B, pa, pb, out):
    """The VLM's change claims, segmented and checked in 3D (see check)."""
    claims = ask_vlm(pa, pb, out / "vlm.json")
    return check(claim_objects(claims, segment(claims, {"A": pa, "B": pb})), A, B, pa, pb, out) if claims else []


def claim_objects(claims, masks):
    """VLM claims as candidate objects: {name, change, description, score, masks: {(video, frame): mask}}."""
    return [{"name": o["name"], "change": o["change"], "description": o["description"], "score": float(o["confidence"]),
             "masks": {(v, f): m for (kk, v, f), m in masks.items() if kk == k}} for k, o in enumerate(claims)]


def check(objects, A, B, pa, pb, out):
    """Each candidate object, lifted to 3D and tested against the other video, as a change with label, sentence,
    type, status, evidence, 3D box and points, and a before/after crop. An object may carry a "veto" (a reason it
    is rejected before any geometry, e.g. the VLM saw no change there)."""
    views, frames = {"A": A, "B": B}, {"A": pa, "B": pb}
    feats = {"A": change2d.dino(A.images), "B": change2d.dino(B.images)}
    room = np.concatenate([recon.cloud(A)[0], recon.cloud(B)[0]])
    lo_room, hi_room = np.percentile(room, 2, axis=0), np.percentile(room, 98, axis=0)
    changes = []
    for k, o in enumerate(objects):
        masks = o["masks"]
        pts, fts, rpts, rfts, best = {}, {}, {}, {}, {}
        for v in "AB":
            got = [(f, *lift(m, views[v], f, feats[v]), m.sum()) for (vv, f), m in masks.items() if vv == v]
            if got and sum(len(g[1]) for g in got) >= 20:
                cat = lambda n: np.concatenate([g[n] for g in got])
                pts[v], fts[v], rpts[v], rfts[v] = cat(1), cat(2), cat(3), cat(4)
                best[v] = max(got, key=lambda g: g[5])[0]
        kind = o["change"]
        if kind == "added" and "B" not in pts or kind == "removed" and "A" not in pts or not pts:
            continue  # no usable mask where the claim needs one
        if kind == "moved" and len(pts) < 2:
            kind = "added" if "B" in pts else "removed"
        here = "B" if kind in ("added", "moved") else "A"
        other = "A" if here == "B" else "B"
        ev = [evidence(pts[here], fts[here], views[other], feats[other])]
        ctx = [evidence(rpts[here], rfts[here], views[other], feats[other])]
        shift = None
        if kind == "moved":
            shift = float(np.linalg.norm(np.median(pts["A"], 0) - np.median(pts["B"], 0)))
        area = np.mean([m.mean() for (vv, _), m in masks.items() if vv == here])
        status, why = ("rejected", o["veto"]) if o.get("veto") else judge(kind, ev, ctx, shift, area)
        box = lambda P: (np.percentile(P, 5, 0).tolist(), np.percentile(P, 95, 0).tolist())
        lo, hi = box(pts[here])
        c = {"type": kind, "label": o["name"], "description": o["description"], "status": status, "why": why,
             "score": float(o["score"]), "source": o.get("source", "vlm"), "lo": lo, "hi": hi, "from_lo": None,
             "from_hi": None, "evidence": ev[0], "context": ctx[0], "support": support(ev[0], ctx[0]), "shift_m": shift,
             "where": where(np.median(pts[here], 0), lo_room, hi_room),
             "points": {v: pts[v][::max(1, len(pts[v]) // 3000)].tolist() for v in pts},
             "frames": {v: sorted(f for (vv, f) in masks if vv == v) for v in pts}}
        if kind == "moved":
            c["from_lo"], c["from_hi"] = box(pts["A"])
        c["text"] = sentence(c)
        crops(c, k, best, masks, frames, views, out)
        changes.append(c)
    order = {"confirmed": 0, "unverified": 1, "rejected": 2}
    return sorted(changes, key=lambda c: (order[c["status"]], -c["score"]))


def where(p, lo, hi):
    import label
    return label.where(p, lo, hi)


def sentence(c):
    s = {"added": "Added", "removed": "Removed", "moved": "Moved"}[c["type"]] + f": {c['label']}"
    if c["type"] == "moved" and c["shift_m"] is not None:
        s += f", {c['shift_m']:.1f} m"
    return f"{s}, {c['where']}." + (f" {c['description']}" if c["description"] else "")


def crops(c, k, best, masks, frames, views, out):
    """Before/after picture: the object's best mask in its own video, and the same 3D spot in the other video."""
    d = out / "changes"; d.mkdir(exist_ok=True)
    tiles = []
    for v in "AB":
        if v in best:
            im = np.asarray(Image.open(frames[v][best[v]]).convert("RGB")).copy()
            m = masks[(v, best[v])]
            edge = ndimage.binary_dilation(m, iterations=max(2, im.shape[1] // 300)) & ~m  # outline, keeps true colors
            im[edge] = [255, 60, 60] if v == "A" else [60, 230, 90]
            ys, xs = np.nonzero(m)
            box = (xs.min(), ys.min(), xs.max(), ys.max())
        else:  # project the object's points into the other video's frame that sees most of them
            P = np.array(c["points"]["B" if v == "A" else "A"])
            vw = views[v]
            H, W = vw.depth.shape[1:]
            best_j, best_n, best_uv = None, 0, None
            for j in range(len(vw.E)):
                q = P @ vw.E[j, :, :3].T + vw.E[j, :, 3]
                uv = (q @ vw.K[j].T)[:, :2] / np.maximum(q[:, 2:], 1e-6)
                ok = (q[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                if ok.sum() > best_n:
                    best_j, best_n, best_uv = j, ok.sum(), uv[ok]
            if best_j is None:
                continue
            im = np.asarray(Image.open(frames[v][best_j]).convert("RGB"))
            s = im.shape[1] / W
            (x0, y0), (x1, y1) = best_uv.min(0) * s, best_uv.max(0) * s
            box = (x0, y0, x1, y1)
        x0, y0, x1, y1 = box
        p = 0.6 * max(x1 - x0, y1 - y0) + 20
        crop = Image.fromarray(im).crop((max(0, x0 - p), max(0, y0 - p), min(im.shape[1], x1 + p), min(im.shape[0], y1 + p)))
        crop.thumbnail((360, 360))
        tiles.append((v, crop))
    if not tiles:
        return
    W = sum(t.width for _, t in tiles) + 10 * (len(tiles) - 1)
    sheet = Image.new("RGB", (W, max(t.height for _, t in tiles) + 20), "white")
    x, dr = 0, ImageDraw.Draw(sheet)
    for v, t in tiles:
        sheet.paste(t, (x, 20)); dr.text((x + 4, 4), "before (A)" if v == "A" else "after (B)", fill="black")
        x += t.width + 10
    c["image"] = f"changes/{k:02d}.jpg"
    sheet.save(out / c["image"], quality=88)
