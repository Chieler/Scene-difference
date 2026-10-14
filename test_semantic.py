import numpy as np
from recon import Views
from semantic import evidence, judge, support, MIN_AREA


def camera(depth, S=2, H=28, W=28):
    K = np.tile(np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]]), (S, 1, 1))
    E = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (S, 1, 1))
    return Views(np.zeros((S, H, W, 3), np.uint8), np.broadcast_to(depth, (S, H, W)).copy(), np.ones((S, H, W)), K, E)


def unit(rng, n, c=16):
    f = rng.normal(size=(n, c)); return f / np.linalg.norm(f, axis=-1, keepdims=True)


X = np.array([[x, y, 2.0] for x in (-0.05, 0, 0.05) for y in (-0.05, 0, 0.05)])  # object 2 m in front


def test_other_video_seeing_through_the_spot_is_free_space():
    rng = np.random.default_rng(0)
    fx = unit(rng, len(X))
    fo = np.broadcast_to(fx[0], (2, 28, 28, 16)).astype(np.float16)
    e = evidence(X, fx, camera(3.0), fo)  # the other video sees a wall 1 m behind where the object stood
    assert e["seen"] == 1.0 and e["gone"] == 1.0


def test_something_nearer_in_the_way_means_never_seen():
    rng = np.random.default_rng(0)
    e = evidence(X, unit(rng, len(X)), camera(1.0), np.zeros((2, 28, 28, 16), np.float16))
    assert e["seen"] == 0.0


def test_same_surface_same_look_is_no_evidence():
    rng = np.random.default_rng(0)
    fx = unit(rng, 1).repeat(len(X), 0)
    fo = np.broadcast_to(fx[0], (2, 28, 28, 16)).astype(np.float16)
    e = evidence(X, fx, camera(2.0), fo)
    assert e["seen"] == 1.0 and e["gone"] == 0.0 and e["differs"] < 1e-2


def test_judge_contrasts_the_object_with_its_background_ring():
    ring = {"seen": 1.0, "gone": 0.3, "differs": 0.4}  # registration noise: the background also looks changed
    assert judge("removed", [{"seen": 1.0, "gone": 0.35, "differs": 0.45}], [ring], None, 0.02)[0] == "rejected"
    assert judge("removed", [{"seen": 1.0, "gone": 0.9, "differs": 0.7}], [ring], None, 0.02)[0] == "confirmed"
    assert support({"gone": 0.3, "differs": 0.7}, {"gone": 0.3, "differs": 0.4}) > 0.5  # appearance alone suffices


def test_judge_keeps_unobserved_and_tiny_claims_as_unverified():
    ring = {"seen": 1.0, "gone": 0.0, "differs": 0.0}
    assert judge("added", [{"seen": 0.1, "gone": 0.0, "differs": 0.0}], [ring], None, 0.02)[0] == "unverified"
    assert judge("added", [{"seen": 1.0, "gone": 0.0, "differs": 0.0}], [ring], None, MIN_AREA / 2)[0] == "unverified"


def test_judge_rejects_a_move_that_stayed_in_place():
    assert judge("moved", [], [], 0.05, 0.02)[0] == "rejected"
    assert judge("moved", [], [], 0.8, 0.02)[0] == "confirmed"
