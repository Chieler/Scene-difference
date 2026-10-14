import numpy as np
from change2d import scores, vote
from changes import Grid
from recon import Views


def flat_wall(S=2, H=28, W=28):
    K = np.tile(np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]]), (S, 1, 1))
    E = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (S, 1, 1))
    return Views(np.zeros((S, H, W, 3), np.uint8), np.full((S, H, W), 2.0), np.ones((S, H, W)), K, E)


def test_only_the_changed_patches_score_high_and_vote_into_one_blob():
    rng = np.random.default_rng(0)
    fs = rng.normal(size=(2, 14, 14, 64)); fs /= np.linalg.norm(fs, axis=-1, keepdims=True)
    fd = fs.copy()
    fd[:, 6:9, 6:9] = -fs[:, 6:9, 6:9]  # same spot, different look in the other video
    P, s = scores(flat_wall(), flat_wall(), fs.astype(np.float16), fd.astype(np.float16))
    s = s.reshape(2, 14, 14)
    assert (s[:, 7, 7] > 0.5).all()  # center of the changed block: only random neighbors in the window
    assert np.abs(s[:, :4, :4]).max() < 1e-2  # far from it: identical
    bl = vote(P, s.reshape(-1), Grid.around(P), "removed", thr=0.5, min_voxels=1)
    assert len(bl) == 1 and abs(bl[0].center[:2]).max() < 0.15  # the block sits at the image center
