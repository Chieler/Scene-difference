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
    small, large = (oa, ob) if oa.sum() <= ob.sum() else (ob, oa)
    near = ndimage.binary_dilation(large, iterations=2)
    return float((small & near).sum() / max(1, small.sum()))
