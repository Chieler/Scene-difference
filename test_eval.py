import numpy as np
from pycocotools import mask as rle
from eval import project_rect, iou, gt_rects, score

K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
E = np.hstack([np.eye(3), np.zeros((3, 1))])  # camera at origin looking down +z


def test_project_rect_and_iou():
    r = project_rect([-0.1, -0.1, 1.0], [0.1, 0.1, 1.0], K, E, (100, 100))
    assert np.allclose(r, (40, 40, 60, 60))
    assert project_rect([-0.1, -0.1, -1.0], [0.1, 0.1, 1.0], K, E, (100, 100)) is None  # behind camera
    assert iou(r, r) == 1.0 and iou(r, (0, 0, 10, 10)) == 0.0 and iou(r, None) == 0.0


def test_gt_rects_scales_to_our_image():
    m = np.zeros((200, 100), np.uint8, order="F")
    m[20:40, 10:30] = 1
    enc = rle.encode(m)
    enc["counts"] = enc["counts"].decode()  # SceneDiff stores counts as str
    r = gt_rects({"0": enc}, 2, 4, (100, 50))  # frame 0 -> native 0, frame 1 -> native 2 (missing)
    assert np.allclose(r[0], (5, 10, 15, 20)) and r[1] is None


def test_score_counts_moved_split_as_found_but_wrong_type():
    preds = [{"type": "removed"}, {"type": "added"}, {"type": "added"}]
    gts = [{"type": "moved"}, {"type": "added"}]
    sim = np.array([[0.5, 0.0], [0.4, 0.0], [0.0, 0.1]])
    assert score(preds, gts, sim) == {"preds": 3, "tp": 2, "gts": 2, "found": 1, "typed": 0}
