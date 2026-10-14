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

def test_overlap_bounded_and_symmetric_when_b_smaller():
    a = scan([ROOM, STATIC], CAMS_A)
    b = scan([ROOM], CAMS_B)
    g = Grid.around(a[0], b[0])
    sa, sb = carve(g, *a), carve(g, *b)
    assert (sb == OCC).sum() < (sa == OCC).sum()
    assert overlap(sa, sb) <= 1.0
    assert overlap(sa, sb) == overlap(sb, sa)
