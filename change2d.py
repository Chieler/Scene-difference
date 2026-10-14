# change2d.py
"""Image-space change detection: compare each frame's DINOv2 patches with the same 3D spot seen from the other
video's nearest-pose frames, then vote in 3D. Catches small objects on surfaces that the voxel diff erases."""
import numpy as np
from scipy import ndimage
from changes import Blob, VOXEL

DINO = "facebook/dinov2-base"
UPSCALE = 2  # DINO patches are 14 px; at 2x a 518 px frame gives 74 patches across
NEAREST = 8  # other-video frames compared per frame, by camera pose
OCC_TOL = 0.2  # other view's depth this much (relative) in front of the point: occluded there, no evidence
MAX_DEPTH = 8.0  # meters; beyond this VGGT depth is too loose to reproject

_m = None


def dino(images):
    """(S, H, W, 3) uint8 -> (S, h, w, C) unit-norm float16 patch features."""
    global _m
    import torch
    from transformers import AutoModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if _m is None:
        _m = AutoModel.from_pretrained(DINO).to(dev).eval()
    S, H, W, _ = images.shape
    h, w = H * UPSCALE // 14, W * UPSCALE // 14
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    out = []
    with torch.no_grad():
        for s in range(0, S, 8):
            x = torch.from_numpy(images[s:s + 8]).float().div(255).sub(mean).div(std).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, (h * 14, w * 14), mode="bilinear").to(dev)
            f = _m(pixel_values=x).last_hidden_state[:, 1:].reshape(len(x), h, w, -1)
            out.append(torch.nn.functional.normalize(f, dim=-1).half().cpu().numpy())
    return np.concatenate(out)


def nearest_frames(src, dst, k=NEAREST):
    """For each src frame, the k dst frames with the closest camera position and viewing direction."""
    c = lambda v: -np.einsum("sji,sj->si", v.E[:, :, :3], v.E[:, :, 3])
    d = np.linalg.norm(c(src)[:, None] - c(dst)[None], axis=2) + (1 - src.E[:, 2, :3] @ dst.E[:, 2, :3].T)
    return np.argsort(d, axis=1)[:, :k]


def scores(src, dst, fs, fd):
    """World points at src patch centers and their change score: 1 - best DINO similarity with the same spot in
    any nearby dst frame that sees it (3x3 patch window for reprojection slack). NaN where no dst frame sees it."""
    S, H, W = src.depth.shape
    _, h, w, _ = fs.shape
    yi = ((np.arange(h) + 0.5) * H / h).astype(int)
    xi = ((np.arange(w) + 0.5) * W / w).astype(int)
    vv, uu = np.meshgrid(yi, xi, indexing="ij")
    near = nearest_frames(src, dst)
    pts, out = [], []
    for i in range(S):
        d = src.depth[i][vv, uu]
        K, R, t = src.K[i], src.E[i, :, :3], src.E[i, :, 3]
        cam = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1).reshape(-1, 3)
        X = (cam - t) @ R
        f = fs[i].reshape(-1, fs.shape[-1]).astype(np.float32)
        best = np.full(len(X), np.nan, np.float32)
        for j in near[i]:
            c = X @ dst.E[j, :, :3].T + dst.E[j, :, 3]
            z = c[:, 2]
            uv = c @ dst.K[j].T
            u, v = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
            ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ok[ok] &= dst.depth[j][v[ok].astype(int), u[ok].astype(int)] > z[ok] * (1 - OCC_TOL)
            if not ok.any():
                continue
            pa, pb = (v[ok] * h / H).astype(int), (u[ok] * w / W).astype(int)
            sim = np.full(ok.sum(), -1.0, np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    g = fd[j][np.clip(pa + dy, 0, h - 1), np.clip(pb + dx, 0, w - 1)].astype(np.float32)
                    sim = np.maximum(sim, np.einsum("nc,nc->n", g, f[ok]))
            best[ok] = np.fmax(best[ok], sim)
        keep = (d.reshape(-1) > 0) & (d.reshape(-1) < MAX_DEPTH)
        pts.append(X[keep]); out.append(1 - best[keep])
    return np.concatenate(pts).astype(np.float32), np.concatenate(out)


def vote(P, s, grid, kind, thr=0.5, ratio=0.5, min_votes=2, min_voxels=5):
    """Voxels where most observations call it changed; connected (1-voxel gaps bridged) into blobs."""
    idx = grid.index(P); ok = grid.inside(idx) & ~np.isnan(s)
    flat = np.ravel_multi_index(idx[ok].T, grid.shape)
    n = int(np.prod(grid.shape))
    tot = np.bincount(flat, minlength=n)
    chg = np.bincount(flat[s[ok] > thr], minlength=n)
    mask = ((chg >= min_votes) & (chg >= ratio * np.maximum(tot, 1))).reshape(grid.shape)
    lab, _ = ndimage.label(ndimage.binary_dilation(mask), structure=np.ones((3, 3, 3)))
    lab[~mask] = 0
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        b = np.argwhere(lab[sl] == i) + [x.start for x in sl]
        if len(b) < min_voxels:
            continue
        c = grid.center(b)
        lo, hi = c.min(0) - VOXEL / 2, c.max(0) + VOXEL / 2
        out.append(Blob(kind, b, lo, hi, (lo + hi) / 2, len(b) * VOXEL**3))
    return out
