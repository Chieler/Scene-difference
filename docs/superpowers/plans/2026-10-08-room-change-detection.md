# Room Change Detection MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI that takes two RGB phone videos of one room and reports (text + 3D HTML viewer) which objects were added, removed or moved.

**Architecture:** VGGT reconstructs frames from both videos in one forward pass, so poses and points share one frame. The frame is rotated gravity-up and scaled to meters from camera height. Each video is carved into a 5 cm voxel grid (occupied / free / unknown). The diff only reports voxels that are occupied in one video and known free in the other, so unseen regions never count as changes. Changed blobs are labeled with CLIP zero-shot, and a removed blob and an added blob with similar appearance and volume pair into "moved".

**Tech Stack:** Python 3.12, uv, numpy, scipy, torch (MPS), VGGT (`facebook/VGGT-1B`), sentence-transformers (`clip-ViT-B-32`), ffmpeg, three.js (CDN) for the viewer.

**Spec:** `docs/superpowers/specs/2026-10-08-room-change-detection-design.md`

**Note:** the project directory is not a git repo, so there are no commit steps. Run `git init` first if you want them.

## Global Constraints

- Input is plain RGB video. No LiDAR. No VLM or API calls. After the model downloads, everything runs offline.
- Voxel size 5 cm (`changes.VOXEL = 0.05`). Minimum change size about 10 cm.
- Metric scale comes from a fixed assumed camera height of 1.4 m (`recon.CAM_HEIGHT`). Sizes are not reported in the MVP; the scale only serves the voxel size.
- One command: `uv run diff.py A.mp4 B.mp4 -o out/`.
- Outputs: `out/viewer.html`, `out/report.md`, `out/changes.json`.
- Files: `diff.py`, `recon.py`, `changes.py`, `label.py`, `viewer.py`, `test_changes.py`, `test_recon.py`, `README.md`, `scripts/spike_vggt.py`.
- Deferred (do not build): `--sizes`, `--vocab`, crop gallery, MASt3R fallback, sub-10 cm changes, GUI, server, tracking.

## Review Focus

1. Identical videos (no changes): report says "No changes detected" and the viewer still builds (Task 2 test, Task 6).
2. A region only one video saw must never be reported as added or removed (Task 2 test).
3. A wall with more points than the floor must not be picked as the floor (Task 3 test).
4. Videos that do not overlap must stop with a clear message and produce no diff (Task 2 test, Task 6).
5. A person or pet walking through a video shows up as added or removed. "person" is in the vocabulary so it is labeled honestly (Task 4).
6. Missing video file, or portrait video next to a landscape video: clear error, not a traceback (Task 6).

---

### Task 1: Project setup and model spike

**Files:**
- Create: `pyproject.toml` (via uv), `scripts/spike_vggt.py`

**Interfaces:**
- Produces: a working `uv run` environment; measured max frame count for VGGT on this machine (feeds the `--frames` default in Task 6).

- [ ] **Step 1: Create the environment**

```bash
cd /Users/chielerli/Programming/humanoid_test
uv init --bare --python 3.12
uv add numpy scipy pillow torch torchvision sentence-transformers "vggt @ git+https://github.com/facebookresearch/vggt.git"
uv add --dev pytest
```

Expected: `pyproject.toml` and `uv.lock` exist. If the `vggt` install fails (no `pyproject.toml` in that repo), clone it to `third_party/vggt`, `uv add --editable third_party/vggt`, and note this in the README.

- [ ] **Step 2: Write the spike**

```python
# scripts/spike_vggt.py
"""Throwaway: does VGGT load and run on MPS, and how many frames fit?"""
import sys, time, torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
for s in map(int, sys.argv[1:] or ["16", "32", "48"]):
    x = torch.rand(s, 3, 294, 518, device=dev)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    ext, intr = pose_encoding_to_extri_intri(out["pose_enc"], x.shape[-2:])
    print(s, "frames OK", f"{time.time() - t0:.1f}s", {k: tuple(v.shape) for k, v in out.items()}, ext.shape, intr.shape)
```

- [ ] **Step 3: Run it**

Run: `uv run scripts/spike_vggt.py 16 32 48`
Expected: one "frames OK" line per size with shapes. `depth` should be `(1, S, 294, 518, 1)`, `depth_conf` `(1, S, 294, 518)`, `ext` `(1, S, 3, 4)`, `intr` `(1, S, 3, 3)`. If these differ, fix the indexing in Task 3's `run_vggt` to match. If a size runs out of memory, that size is the ceiling. Set the `--frames` default (frames per video) to half the largest size that passed. Delete `scripts/spike_vggt.py` afterwards only if you do not want to keep it. It is labeled throwaway.

---

### Task 2: Free-space change detection (`changes.py`)

**Files:**
- Create: `changes.py`, `test_changes.py`

**Interfaces:**
- Produces:
  - `VOXEL: float`, `UNKNOWN, FREE, OCC: int`
  - `Grid.around(*clouds) -> Grid`, with `index(p) -> int[N,3]`, `inside(idx) -> bool[N]`, `center(idx) -> float[N,3]`, fields `lo`, `shape`
  - `carve(grid, points, origins, min_hits=2) -> uint8[shape]` (points and origins are `float[N,3]`, one origin per point)
  - `diff(a, b, tol=1) -> (added_mask, removed_mask)` (bool grids)
  - `Blob(kind, idx, lo, hi, center, volume)` and `blobs(mask, kind, grid, min_voxels=20) -> list[Blob]`
  - `pair_moved(removed, added, sim, min_sim=0.85, vol_ratio=0.5) -> list[(i_removed, j_added)]`; `sim` is a `[len(removed), len(added)]` matrix
  - `overlap(a, b) -> float` in 0..1

- [ ] **Step 1: Write the failing tests**

```python
# test_changes.py
import numpy as np
from changes import *

def box(lo, hi, step=0.02):
    lo, hi = np.array(lo, float), np.array(hi, float)
    g = [np.linspace(l, h, round((h - l) / step) + 1) for l, h in zip(lo, hi)]
    X = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    on = np.isclose(X, lo, atol=1e-6) | np.isclose(X, hi, atol=1e-6)
    return X[on.any(1)]

ROOM = np.concatenate([box([0, 0, 0], [4, 4, 0]), box([0, 4, 0], [4, 4, 2.5])])  # floor + back wall
CAMS_A = np.array([[0.2, 0.2, 1.5], [0.4, 3.6, 1.5]])
CAMS_B = np.array([[3.8, 0.2, 1.5], [2.0, 0.1, 1.5]])

def scan(parts, cams, keep=None):
    p = np.concatenate(parts)
    if keep is not None:
        p = p[keep(p)]
    o = cams[np.argmin(np.linalg.norm(p[:, None] - cams[None], axis=2), axis=1)]
    return p.astype(np.float32), o.astype(np.float32)

def run(a, b):
    g = Grid.around(a[0], b[0])
    sa, sb = carve(g, *a), carve(g, *b)
    add, rem = diff(sa, sb)
    return g, sa, sb, blobs(add, "added", g), blobs(rem, "removed", g)

STATIC = box([2.5, 2.5, 0], [2.9, 2.9, 0.4])
GONE = box([1, 1, 0], [1.4, 1.4, 0.4])
NEW = box([3, 1, 0], [3.4, 1.4, 0.4])

def test_added_and_removed():
    a = scan([ROOM, STATIC, GONE], CAMS_A)
    b = scan([ROOM, STATIC, NEW], CAMS_B)
    _, _, _, add, rem = run(a, b)
    assert len(add) == 1 and len(rem) == 1
    assert np.linalg.norm(add[0].center - [3.2, 1.2, 0.2]) < 0.2
    assert np.linalg.norm(rem[0].center - [1.2, 1.2, 0.2]) < 0.2

def test_no_change_gives_no_blobs():
    a = scan([ROOM, STATIC], CAMS_A)
    b = scan([ROOM, STATIC], CAMS_B)
    _, _, _, add, rem = run(a, b)
    assert add == [] and rem == []

def test_unseen_region_is_not_a_change():
    cams = np.array([[0.2, 0.2, 1.5], [0.2, 3.6, 1.5]])
    a = scan([ROOM, STATIC], cams, keep=lambda p: p[:, 0] < 2.5)  # A never looks at x > 2.5
    b = scan([ROOM, STATIC, NEW], CAMS_B)
    g, sa, _, add, _ = run(a, b)
    assert sa[tuple(g.index(np.array([[3.2, 1.2, 0.2]]))[0])] == UNKNOWN
    assert not any(np.linalg.norm(x.center - [3.2, 1.2, 0.2]) < 0.3 for x in add)

def test_pair_moved_respects_similarity_and_volume():
    def fake(vox):
        idx = np.zeros((vox, 3), int)
        return Blob("x", idx, np.zeros(3), np.ones(3), np.full(3, .5), vox * VOXEL**3)
    rem, add = [fake(100), fake(100)], [fake(100), fake(5)]
    sim = np.array([[0.95, 0.99], [0.2, 0.99]])
    assert pair_moved(rem, add, sim) == [(0, 0)]  # (1,1) rejected: volume ratio; (0,1) rejected: volume ratio

def test_overlap_high_for_same_room_low_for_disjoint():
    a = scan([ROOM], CAMS_A)
    b = scan([ROOM], CAMS_B)
    g = Grid.around(a[0], b[0])
    assert overlap(carve(g, *a), carve(g, *b)) > 0.8
    far = scan([box([10, 10, 0], [12, 12, 0])], np.array([[10.5, 10.5, 1.5]]))
    g2 = Grid.around(a[0], far[0])
    assert overlap(carve(g2, *a), carve(g2, *far)) < 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test_changes.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'changes'`.

- [ ] **Step 3: Implement**

```python
# changes.py
"""Free-space-carving diff of two scans in one shared, gravity-up, metric frame."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

VOXEL = 0.05
UNKNOWN, FREE, OCC = 0, 1, 2


@dataclass
class Grid:
    lo: np.ndarray
    shape: tuple

    @classmethod
    def around(cls, *clouds, margin=0.2):
        pts = np.concatenate(clouds)
        lo = np.percentile(pts, 0.5, axis=0) - margin
        hi = np.percentile(pts, 99.5, axis=0) + margin
        return cls(lo, tuple(np.ceil((hi - lo) / VOXEL).astype(int)))

    def index(self, p):
        return np.floor((np.asarray(p) - self.lo) / VOXEL).astype(int)

    def inside(self, idx):
        return np.all((idx >= 0) & (idx < self.shape), axis=1)

    def center(self, idx):
        return self.lo + (np.asarray(idx) + 0.5) * VOXEL


def carve(grid, points, origins, min_hits=2, chunk=20000):
    """OCC: >= min_hits depth points. FREE: a camera ray passed through. UNKNOWN: never seen."""
    hits = np.zeros(int(np.prod(grid.shape)), np.int64)
    seen = np.zeros(grid.shape, bool)
    for s in range(0, len(points), chunk):
        p, o = points[s:s + chunk], origins[s:s + chunk]
        L = np.linalg.norm(p - o, axis=1)
        reach = np.clip(L - VOXEL, 0, None)  # stop one voxel short of the surface: depth noise tolerance
        end = o + (p - o) * (reach / np.maximum(L, 1e-6))[:, None]
        n = int(np.ceil(reach.max() / (VOXEL / 2))) + 1
        t = np.linspace(0, 1, n, dtype=np.float32)[None, :, None]
        idx = grid.index((o[:, None] + (end - o)[:, None] * t).reshape(-1, 3))
        idx = idx[grid.inside(idx)]
        seen[tuple(idx.T)] = True
        idx = grid.index(p)
        idx = idx[grid.inside(idx)]
        hits += np.bincount(np.ravel_multi_index(idx.T, grid.shape), minlength=hits.size)
    state = np.full(grid.shape, UNKNOWN, np.uint8)
    state[seen] = FREE
    state[hits.reshape(grid.shape) >= min_hits] = OCC
    return state


def diff(a, b, tol=1):
    """added: OCC in b and known FREE in a (not merely unseen). removed: the mirror."""
    near_a = ndimage.binary_dilation(a == OCC, iterations=tol)
    near_b = ndimage.binary_dilation(b == OCC, iterations=tol)
    return (b == OCC) & (a == FREE) & ~near_a, (a == OCC) & (b == FREE) & ~near_b


@dataclass
class Blob:
    kind: str
    idx: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    center: np.ndarray
    volume: float


def blobs(mask, kind, grid, min_voxels=20):
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        idx = np.argwhere(lab[sl] == i) + [s.start for s in sl]
        if len(idx) < min_voxels:
            continue
        c = grid.center(idx)
        lo, hi = c.min(0) - VOXEL / 2, c.max(0) + VOXEL / 2
        out.append(Blob(kind, idx, lo, hi, (lo + hi) / 2, len(idx) * VOXEL**3))
    return out


def pair_moved(removed, added, sim, min_sim=0.85, vol_ratio=0.5):
    """Greedy best-first pairing of a removed and an added blob that look alike and have similar volume."""
    cands = sorted(((sim[i][j], i, j) for i in range(len(removed)) for j in range(len(added))), reverse=True)
    pairs, used_r, used_a = [], set(), set()
    for s, i, j in cands:
        if s < min_sim:
            break
        v1, v2 = removed[i].volume, added[j].volume
        if i in used_r or j in used_a or min(v1, v2) / max(v1, v2) < vol_ratio:
            continue
        pairs.append((i, j)); used_r.add(i); used_a.add(j)
    return pairs


def overlap(a, b):
    """Share of the smaller scan's occupied voxels that the other scan also occupies (within 2 voxels)."""
    oa, ob = a == OCC, b == OCC
    near = ndimage.binary_dilation(ob, iterations=2)
    return float((oa & near).sum() / max(1, min(oa.sum(), ob.sum())))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test_changes.py -q`
Expected: 5 passed. These tests exercise thresholds (`min_hits`, `tol`, `min_voxels`) against a synthetic room. If `test_no_change_gives_no_blobs` shows stray blobs, raise `tol` to 2 or `min_voxels` to 30 in `changes.py`, never by editing the tests. The tuned values are what the real videos will use as defaults.

---

### Task 3: Reconstruction and gravity alignment (`recon.py`)

**Files:**
- Create: `recon.py`, `test_recon.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (the changes module is not imported here).
- Produces:
  - `CAM_HEIGHT = 1.4`
  - `Views(images, depth, conf, K, E)` dataclass. `images uint8[S,H,W,3]`, `depth float[S,H,W]`, `conf float[S,H,W]`, `K float[S,3,3]`, `E float[S,3,4]` (world to camera). Method `part(a, b) -> Views`.
  - `centers(v) -> float[S,3]` (camera centers in world).
  - `cloud(v, conf_pct=40, stride=3, max_pts=150_000) -> (points float32[N,3], origins float32[N,3], colors uint8[N,3])`
  - `floor_transform(points, cams, ups) -> (s, R, p0)` such that `w' = s * R @ (w - p0)` is gravity-up with the floor at z = 0 and camera height `CAM_HEIGHT`.
  - `align(v, s, R, p0) -> Views` (new `E` and `depth`, same images).
  - `reconstruct(video_a, video_b, frames, work_dir) -> (Views_a, Views_b)` (aligned).

- [ ] **Step 1: Write the failing tests**

```python
# test_recon.py
import numpy as np
from recon import floor_transform, CAM_HEIGHT

def test_floor_wins_over_bigger_wall_and_heights_are_metric():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 4, 3000), rng.uniform(0, 4, 3000), np.zeros(3000)]
    wall = np.c_[rng.uniform(0, 4, 9000), np.full(9000, 4.0), rng.uniform(0, 2.5, 9000)]  # 3x more points
    cams = np.array([[1.0, 1.0, 1.5], [3.0, 1.0, 1.6], [2.0, 2.5, 1.4]])
    # arbitrary rotation, scale, shift: the unknown frame VGGT hands back
    a = 0.7; R0 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    b = 0.5; R1 = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R0 = R1 @ R0
    f = lambda x: 3.7 * x @ R0.T + [5, -2, 1]
    pts, cm, up = f(np.r_[floor, wall]), f(cams), np.tile(R0 @ [0, 0, 1], (3, 1))
    s, R, p0 = floor_transform(pts, cm, up)
    tp = s * (pts[:3000] - p0) @ R.T
    tc = s * (cm - p0) @ R.T
    assert np.abs(tp[:, 2]).max() < 0.05          # floor lies at z = 0
    assert abs(np.median(tc[:, 2]) - CAM_HEIGHT) < 0.05  # cameras at assumed height
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test_recon.py -q`
Expected: `ModuleNotFoundError: No module named 'recon'`.

- [ ] **Step 3: Implement**

```python
# recon.py
"""Videos -> one shared, gravity-up, metric-scale reconstruction (VGGT)."""
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

CAM_HEIGHT = 1.4  # assumed handheld camera height in meters; sets the metric scale
LONG_SIDE = 518


@dataclass
class Views:
    images: np.ndarray
    depth: np.ndarray
    conf: np.ndarray
    K: np.ndarray
    E: np.ndarray

    def part(self, a, b):
        return Views(*(x[a:b] for x in (self.images, self.depth, self.conf, self.K, self.E)))


def centers(v):
    return -np.einsum("sji,sj->si", v.E[:, :, :3], v.E[:, :, 3])


def unproject(depth, K, E):
    H, W = depth.shape
    vv, uu = np.mgrid[:H, :W]
    cam = np.stack([(uu - K[0, 2]) / K[0, 0] * depth, (vv - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return (cam - E[:, 3]) @ E[:, :3]  # R^T (cam - t)


def cloud(v, conf_pct=40, stride=3, max_pts=150_000):
    S, H, W = v.depth.shape
    P = np.stack([unproject(v.depth[i], v.K[i], v.E[i]) for i in range(S)])
    O = np.broadcast_to(centers(v)[:, None, None, :], P.shape)
    keep = (v.conf >= np.percentile(v.conf, conf_pct)) & (v.depth > 0)
    sub = np.zeros_like(keep); sub[:, ::stride, ::stride] = True
    keep &= sub
    P, O, C = P[keep], O[keep], v.images[keep]
    if len(P) > max_pts:
        sel = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
        P, O, C = P[sel], O[sel], C[sel]
    return P.astype(np.float32), O.astype(np.float32), C


def rot_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        raise ValueError("degenerate up vector")
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def floor_transform(points, cams, ups, iters=400, seed=0):
    """RANSAC the largest near-horizontal plane (horizontal = normal close to mean camera up)."""
    rng = np.random.default_rng(seed)
    up0 = ups.mean(0); up0 /= np.linalg.norm(up0)
    pts = points[rng.choice(len(points), min(len(points), 30000), replace=False)]
    thr = 0.02 * np.median(np.linalg.norm(pts[:, None] - cams[None], axis=2))  # scale-free tolerance
    best, best_n = None, 0
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up0) < 0.9:  # more than ~25 degrees from horizontal: a wall, skip
            continue
        inl = np.abs((pts - p[0]) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < 0.05 * len(pts):
        raise RuntimeError("no floor found: keep the floor in view in both videos")
    q = pts[best]; p0 = q.mean(0)
    n = np.linalg.svd(q - p0)[2][-1]
    if n @ up0 < 0:
        n = -n
    h = np.median((cams - p0) @ n)
    if h <= 0:
        raise RuntimeError("cameras are below the detected floor: floor detection failed")
    return CAM_HEIGHT / h, rot_between(n, np.array([0.0, 0.0, 1.0])), p0


def align(v, s, R, p0):
    Rc = v.E[:, :, :3]
    t = s * (v.E[:, :, 3] + np.einsum("sij,j->si", Rc, p0))
    E = np.concatenate([Rc @ R.T, t[:, :, None]], axis=2)
    return Views(v.images, v.depth * s, v.conf, v.K, E)


def extract_frames(video, n, out):
    out.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={n / dur}",
                    "-frames:v", str(n), str(out / "%03d.jpg")], check=True)
    paths = sorted(out.glob("*.jpg"))
    if len(paths) < 4:
        raise RuntimeError(f"{video}: could not extract enough frames")
    return paths


def load_frames(paths, size):
    ims = [np.asarray(Image.open(p).convert("RGB").resize(size, Image.BICUBIC)) for p in paths]
    return np.stack(ims)


def frame_size(path):
    w, h = Image.open(path).size
    s = LONG_SIDE / max(w, h)
    return tuple(max(14, round(d * s / 14) * 14) for d in (w, h))  # VGGT patch size is 14


def run_vggt(images):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.no_grad():
        out = model(x)
    E, K = pose_encoding_to_extri_intri(out["pose_enc"], x.shape[-2:])
    g = lambda t: t[0].float().cpu().numpy()
    return g(out["depth"])[..., 0], g(out["depth_conf"]), g(K), g(E)


def reconstruct(video_a, video_b, frames, work_dir):
    pa = extract_frames(video_a, frames, work_dir / "A")
    pb = extract_frames(video_b, frames, work_dir / "B")
    sa, sb = frame_size(pa[0]), frame_size(pb[0])
    if sa != sb:
        raise RuntimeError(f"videos have different orientation or aspect ({sa} vs {sb}): record both the same way")
    imgs = load_frames(pa + pb, sa)
    depth, conf, K, E = run_vggt(imgs)
    v = Views(imgs, depth, conf, K, E)
    pts, _, _ = cloud(v)
    ups = -E[:, 1, :3]  # camera y points down, so up = -row 1 of R
    s, R, p0 = floor_transform(pts, centers(v), ups)
    v = align(v, s, R, p0)
    return v.part(0, len(pa)), v.part(len(pa), len(pa) + len(pb))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test_recon.py -q`
Expected: 1 passed. `reconstruct` and `run_vggt` are exercised in Task 6 on real videos (they need the model).

---

### Task 4: Labels and wording (`label.py`)

**Files:**
- Create: `label.py`

**Interfaces:**
- Consumes: `Views` and `Blob` fields (`lo`, `hi`, `center`), no imports needed (duck-typed).
- Produces:
  - `best_crop(blob, views) -> PIL.Image | None`
  - `embed(images) -> float[N,512]` (L2-normalised CLIP image embeddings; a `None` image gives a zero row)
  - `name(emb) -> (word, prob)`
  - `where(center, lo, hi) -> str`
  - `sentence(kind, word, place, from_place=None) -> str`

- [ ] **Step 1: Implement**

```python
# label.py
"""CLIP zero-shot naming of changed blobs, plus the report wording."""
import itertools
import numpy as np
from PIL import Image

VOCAB = """chair, armchair, sofa, couch, table, desk, coffee table, side table, stool, bench, bed, pillow, blanket,
cushion, shelf, bookshelf, cabinet, drawer, wardrobe, box, cardboard box, bag, backpack, suitcase, basket, bin,
trash can, lamp, floor lamp, monitor, laptop, keyboard, television, speaker, plant, vase, book, books, bottle,
cup, mug, plate, bowl, pot, shoes, jacket, clothes, towel, rug, carpet, mat, curtain, mirror, painting, clock,
toy, ball, guitar, fan, heater, vacuum cleaner, ladder, bicycle, person, dog, cat, cable, papers, whiteboard"""
VOCAB = [w.strip() for w in VOCAB.replace("\n", " ").split(",") if w.strip()]

_m = None
_txt = None


def _model():
    global _m, _txt
    if _m is None:
        from sentence_transformers import SentenceTransformer
        _m = SentenceTransformer("clip-ViT-B-32")
        _txt = _m.encode([f"a photo of a {w}" for w in VOCAB], normalize_embeddings=True)
    return _m


def embed(images):
    m = _model()
    out = np.zeros((len(images), _txt.shape[1]), np.float32)
    ok = [i for i, im in enumerate(images) if im is not None]
    if ok:
        out[ok] = m.encode([images[i] for i in ok], normalize_embeddings=True)
    return out


def name(emb):
    if not emb.any():
        return "object", 0.0
    logits = 100 * _txt @ emb
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(p.argmax())
    return VOCAB[i], float(p[i])


def best_crop(blob, views, pad=0.15):
    """Largest fully-visible, unoccluded view of the blob's bounding box."""
    corners = np.array(list(itertools.product(*zip(blob.lo, blob.hi))))
    tol = np.linalg.norm(blob.hi - blob.lo) / 2 + 0.1  # a surface nearer than this is in front of the blob
    best, best_area = None, 0
    for i in range(len(views.E)):
        R, t, K = views.E[i, :, :3], views.E[i, :, 3], views.K[i]
        cam, c = corners @ R.T + t, blob.center @ R.T + t
        if c[2] < 0.3 or (cam[:, 2] < 0.1).any():
            continue
        uv = (cam @ K.T); uv = uv[:, :2] / uv[:, 2:]
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        H, W = views.depth[i].shape
        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H or (x1 - x0) < 12 or (y1 - y0) < 12:
            continue
        u, v = (K @ c)[:2] / c[2]
        if views.depth[i][int(v), int(u)] < c[2] - tol:  # occluded
            continue
        if (x1 - x0) * (y1 - y0) > best_area:
            px, py = pad * (x1 - x0), pad * (y1 - y0)
            best_area = (x1 - x0) * (y1 - y0)
            best = Image.fromarray(views.images[i]).crop(
                (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)))
    return best


def where(center, lo, hi):
    """Coarse place in the room. Axes are the reconstruction's own, so left/front are arbitrary but consistent."""
    f = (center[:2] - lo[:2]) / np.maximum(hi[:2] - lo[:2], 1e-6)
    pick = lambda x, names: names[min(2, max(0, int(x * 3)))]
    parts = [w for w in (pick(f[0], ("left", "center", "right")), pick(f[1], ("front", "middle", "back")))
             if w not in ("center", "middle")]
    return f"the {', '.join(parts)} of the room" if parts else "the middle of the room"


def sentence(kind, word, place, from_place=None):
    if kind == "moved":
        return f"Moved: {word}, from {from_place} to {place}" if from_place != place else f"Moved: {word}, within {place}"
    return f"{kind.capitalize()}: {word} in {place}"
```

- [ ] **Step 2: Smoke check**

Run: `uv run python -c "import label, numpy as np; from PIL import Image; print(label.name(label.embed([Image.new('RGB',(64,64),(200,30,30))])[0])); print(label.where(np.array([3.5,.5]), np.zeros(3), np.array([4,4,3])), '|', label.sentence('moved','chair','the left of the room','the right of the room'))"`
Expected: prints a vocabulary word with a probability, then `the right, front of the room | Moved: chair, from the right of the room to the left of the room`. The first run downloads CLIP.

---

### Task 5: Viewer (`viewer.py`)

**Files:**
- Create: `viewer.py`

**Interfaces:**
- Produces: `write_viewer(path, data)`, where `data` is:

```python
{"A": {"pos": float32[N,3], "col": uint8[N,3]},      # video A cloud
 "B": {"pos": ..., "col": ..., "gray": uint8[N,3]},  # B cloud; "gray" = col with unseen-in-A points grey
 "ghost": {"pos": float32[M,3]},                     # A points inside removed blobs
 "changes": [{"type": "added|removed|moved", "text": str, "lo": [3], "hi": [3], "from_lo": [3] | None, "from_hi": [3] | None}]}
```

- [ ] **Step 1: Implement**

```python
# viewer.py
"""Self-contained HTML viewer (three.js from CDN; needs internet to load the library)."""
import base64, json
import numpy as np

HTML = """<!doctype html><meta charset=utf-8><title>Room changes</title>
<style>body{margin:0;overflow:hidden;font:13px system-ui;color:#eee;background:#111}
#ui{position:fixed;top:0;left:0;width:290px;max-height:100%;overflow:auto;background:#000a;padding:10px}
#ui div.c{cursor:pointer;padding:4px 0;border-top:1px solid #333}.lb{position:fixed;font-size:11px;pointer-events:none;
text-shadow:0 0 3px #000}</style>
<div id=ui><b>View</b><br>
<label><input type=radio name=m value=diff checked> Diff</label>
<label><input type=radio name=m value=A> Video A</label>
<label><input type=radio name=m value=B> Video B</label>
<p style="color:#999;margin:6px 0">Diff: B cloud, grey = not seen in A (not comparable), red ghost = removed objects as they were in A.
Green = added, red = removed, blue = moved.</p><div id=list></div></div>
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const buf = s => Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer;
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200); cam.up.set(0, 0, 1);
const r = new THREE.WebGLRenderer({antialias: true}); r.setSize(innerWidth, innerHeight); document.body.append(r.domElement);
const ctl = new OrbitControls(cam, r.domElement);
const pts = (pos, col, size = 0.025) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf(pos)), 3));
  if (col) g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buf(col)), 3, true));
  return new THREE.Points(g, new THREE.PointsMaterial(col ? {size, vertexColors: true} : {size, color: 0xff3030}));
};
const sets = {A: [pts(D.A.pos, D.A.col)], B: [pts(D.B.pos, D.B.col)], diff: [pts(D.B.pos, D.B.gray), pts(D.ghost.pos, null, 0.03)]};
const COLOR = {added: 0x30e050, removed: 0xff3030, moved: 0x3090ff};
const overlay = new THREE.Group(); scene.add(overlay);
const box = (lo, hi, c) => new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi)), c);
const mid = (lo, hi) => lo.map((x, i) => (x + hi[i]) / 2);
const labels = [];
D.changes.forEach((c, i) => {
  overlay.add(box(c.lo, c.hi, COLOR[c.type]));
  if (c.from_lo) {
    overlay.add(box(c.from_lo, c.from_hi, COLOR[c.type]));
    overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(...mid(c.from_lo, c.from_hi)), new THREE.Vector3(...mid(c.lo, c.hi))]), new THREE.LineBasicMaterial({color: 0xffff00})));
  }
  const el = document.createElement('div'); el.className = 'lb'; el.textContent = c.text.split(':')[1].split(' in ')[0].split(',')[0];
  el.style.color = '#' + COLOR[c.type].toString(16); document.body.append(el);
  labels.push([el, new THREE.Vector3(c.lo[0], c.lo[1], c.hi[2] + 0.1).lerp(new THREE.Vector3(c.hi[0], c.hi[1], c.hi[2] + 0.1), 0.5)]);
  const d = document.createElement('div'); d.className = 'c'; d.textContent = c.text;
  d.onclick = () => { const m = mid(c.lo, c.hi); ctl.target.set(...m); cam.position.set(m[0] + 1.5, m[1] - 1.5, m[2] + 1.2); };
  document.getElementById('list').append(d);
});
if (!D.changes.length) document.getElementById('list').textContent = 'No changes detected.';
const show = m => { for (const k in sets) sets[k].forEach(p => (p.parent && scene.remove(p))); sets[m].forEach(p => scene.add(p)); };
document.querySelectorAll('input[name=m]').forEach(i => i.onchange = () => show(i.value)); show('diff');
const g = new THREE.Box3().setFromBufferAttribute(sets.B[0].geometry.attributes.position), c0 = g.getCenter(new THREE.Vector3());
ctl.target.copy(c0); cam.position.set(c0.x + 3, c0.y - 3, c0.z + 3);
(function loop() {
  requestAnimationFrame(loop); ctl.update(); r.render(scene, cam);
  labels.forEach(([el, p]) => { const q = p.clone().project(cam); const on = q.z < 1;
    el.style.display = on ? '' : 'none'; el.style.left = (q.x + 1) / 2 * innerWidth + 'px'; el.style.top = (1 - q.y) / 2 * innerHeight + 'px'; });
})();
</script>"""


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def write_viewer(path, data):
    f = lambda d, keys: {k: _b64(d[k], np.float32 if k == "pos" else np.uint8) for k in keys}
    payload = {"A": f(data["A"], ["pos", "col"]), "B": f(data["B"], ["pos", "col", "gray"]),
               "ghost": f(data["ghost"], ["pos"]), "changes": data["changes"]}
    path.write_text(HTML.replace("__DATA__", json.dumps(payload)))
```

- [ ] **Step 2: Smoke check with fake data**

Run:
```bash
uv run python -c "
import numpy as np, pathlib, viewer
p=np.random.rand(500,3).astype('f4'); c=(np.random.rand(500,3)*255).astype('u1')
viewer.write_viewer(pathlib.Path('/tmp/v.html'),{'A':{'pos':p,'col':c},'B':{'pos':p,'col':c,'gray':c},'ghost':{'pos':p[:50]},
 'changes':[{'type':'moved','text':'Moved: chair, from the left of the room to the right of the room','lo':[.1,.1,0],'hi':[.3,.3,.4],'from_lo':[.6,.6,0],'from_hi':[.8,.8,.4]}]})"
open /tmp/v.html
```
Expected: browser shows a point cloud, a blue box pair joined by a yellow line, a "chair" label, and a sidebar entry. Orbit and the Diff / A / B toggle work. Check the browser console is clean.

---

### Task 6: CLI, README, real run (`diff.py`)

**Files:**
- Create: `diff.py`, `README.md`, `examples/` (user's videos and the resulting `out/`)

**Interfaces:**
- Consumes: everything above, exactly as specified in the earlier tasks.
- Produces: the CLI. Exit code 1 and a one-line message on every expected failure.

- [ ] **Step 1: Implement**

```python
# diff.py
"""uv run diff.py A.mp4 B.mp4 -o out/"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from changes import Grid, carve, diff, blobs, pair_moved, overlap, UNKNOWN
from recon import reconstruct, cloud
import label
from viewer import write_viewer

MIN_OVERLAP = 0.25  # below this the videos do not show the same space
FRAMES = 24  # per video; set from the Task 1 spike


def fail(msg):
    sys.exit(f"error: {msg}")


def main():
    ap = argparse.ArgumentParser(description="Find what changed in a room between two videos.")
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--frames", type=int, default=FRAMES, help="frames sampled per video")
    args = ap.parse_args()
    for p in (args.a, args.b):
        if not p.is_file():
            fail(f"video not found: {p}")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        A, B = reconstruct(args.a, args.b, args.frames, args.out / "frames")
    except (RuntimeError, ValueError) as e:
        fail(str(e))
    (pa, oa, ca), (pb, ob, cb) = cloud(A), cloud(B)
    grid = Grid.around(pa, pb)
    sa, sb = carve(grid, pa, oa), carve(grid, pb, ob)
    ov = overlap(sa, sb)
    if ov < MIN_OVERLAP:
        fail(f"videos don't overlap enough ({ov:.0%}), re-record with shared walls and floor in view")

    added, removed = diff(sa, sb)
    ba, br = blobs(added, "added", grid), blobs(removed, "removed", grid)
    ea = label.embed([label.best_crop(b, B) for b in ba])
    er = label.embed([label.best_crop(b, A) for b in br])
    pairs = pair_moved(br, ba, er @ ea.T) if len(ba) and len(br) else []
    lo, hi = grid.lo, grid.lo + np.array(grid.shape) * 0.05
    place = lambda b: label.where(b.center, lo, hi)

    changes, moved_r, moved_a = [], {i for i, _ in pairs}, {j for _, j in pairs}
    box = lambda b: {"lo": b.lo.tolist(), "hi": b.hi.tolist()}
    for i, j in pairs:
        w, pr = label.name(ea[j] + er[i])
        changes.append({"type": "moved", "label": w, "score": pr, **box(ba[j]), "from_lo": br[i].lo.tolist(),
                        "from_hi": br[i].hi.tolist(), "text": label.sentence("moved", w, place(ba[j]), place(br[i]))})
    for kind, bl, em, skip in (("added", ba, ea, moved_a), ("removed", br, er, moved_r)):
        for k, b in enumerate(bl):
            if k not in skip:
                w, pr = label.name(em[k])
                changes.append({"type": kind, "label": w, "score": pr, **box(b), "from_lo": None, "from_hi": None,
                                "text": label.sentence(kind, w, place(b))})

    (args.out / "changes.json").write_text(json.dumps(changes, indent=2))
    (args.out / "report.md").write_text("# What changed\n\n" + (
        "\n".join(f"- {c['text']} (confidence {c['score']:.0%})" for c in changes) or "No changes detected.") + "\n")

    unseen = sa[tuple(np.clip(grid.index(pb), 0, np.array(grid.shape) - 1).T)] == UNKNOWN
    gray = cb.copy(); gray[unseen] = 120
    ghost_mask = np.zeros(grid.shape, bool)
    for b in br:
        ghost_mask[tuple(b.idx.T)] = True
    gi = grid.index(pa); ok = grid.inside(gi)
    ghost = pa[ok][ghost_mask[tuple(gi[ok].T)]]
    write_viewer(args.out / "viewer.html", {"A": {"pos": pa, "col": ca}, "B": {"pos": pb, "col": cb, "gray": gray},
                                            "ghost": {"pos": ghost}, "changes": changes})
    print((args.out / "report.md").read_text())
    print(f"viewer: {args.out / 'viewer.html'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Error-path checks (no model needed)**

Run: `uv run diff.py nope.mp4 also_nope.mp4`
Expected: `error: video not found: nope.mp4`, exit code 1.

- [ ] **Step 3: Real run**

Ask the user to put two videos in `examples/` as `A.mp4` and `B.mp4`: same room, same phone orientation, 20-30 s each, slow pan, floor and walls in view, B filmed from a different spot with 2-3 objects added, removed or moved.

Run: `uv run diff.py examples/A.mp4 examples/B.mp4 -o examples/out`
Expected: `report.md` lists the objects the user changed. Open `examples/out/viewer.html` and compare against what was actually changed. For each wrong or missing item, find the cause before tuning:
- registration failed: overlap message appears (re-record, or check frames).
- false additions at the room edges: raise `diff(..., tol=2)`.
- missing small items: lower `min_voxels` in `blobs`.
- wrong labels: look at the crop with `label.best_crop(...).save(...)` first; the crop may be wrong, not CLIP.

If memory fails, lower `--frames`.

- [ ] **Step 4: README**

Write `README.md` with: what it does (2 lines), setup (`uv sync`, ffmpeg required, first run downloads about 5 GB of model weights), run command, how to record the videos (tips from Step 3), what the output files are, a short design note (joint reconstruction so alignment is free; free-space carving so unseen is not changed; CLIP only labels blobs), and limitations (changes under 10 cm, scale assumes 1.4 m camera height, left/front words follow the reconstruction's own axes, a moving person shows up as a change, viewer needs internet for three.js, CLIP can mislabel odd objects).

- [ ] **Step 5: Full test run**

Run: `uv run pytest -q`
Expected: 6 passed.

---

## Self-review

- **Spec coverage:** pipeline (T3, T2), free-space logic (T2), moved pairing (T2, T6), CLIP labels and sentences (T4), alignment failure message (T2 `overlap`, T6), viewer with A / B / diff toggle and grey unseen (T5, T6), synthetic test (T2), README and examples (T6), spike (T1). The spec's "location relative to nearest labeled static object" is cut down to room thirds (`label.where`); the spec says "or the room extent" so this holds.
- **Types:** `Blob` fields, `Views` fields, `Grid` methods and `viewer` data keys match across tasks. `floor_transform(points, cams, ups)` is called the same way in the test and `reconstruct`.
- **Known unverified:** VGGT output shapes (checked by the Task 1 spike), and the `changes.py` thresholds (tuned by the Task 2 tests and the Task 6 real run). The synthetic test cannot model occlusion; the unseen-region test covers only a never-observed region.
