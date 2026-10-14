import numpy as np
from recon import floor_transform, rot_between, CAM_HEIGHT

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

def test_rot_between_antiparallel():
    z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rot_between(-z, z) @ -z, z)
