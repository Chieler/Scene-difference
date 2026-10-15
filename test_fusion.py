import sys, types
import numpy as np

sys.modules.setdefault("official", types.ModuleType("official"))  # fusion imports it; these tests never call it
from fusion import fuse, iou, kind  # noqa: E402


def box(x0, x1, H=20, W=20):
    m = np.zeros((H, W), bool); m[5:15, x0:x1] = True; return m


def test_kind_follows_which_videos_have_masks():
    assert kind({("A", 0): box(0, 5)}) == "removed"
    assert kind({("B", 3): box(0, 5)}) == "added"
    assert kind({("A", 0): box(0, 5), ("B", 1): box(0, 5)}) == "moved"


def test_iou_only_counts_shared_frames():
    assert iou({("A", 0): box(0, 10)}, {("A", 1): box(0, 10)}) == 0.0
    assert iou({("A", 0): box(0, 10)}, {("A", 0): box(0, 10), ("B", 2): box(5, 9)}) == 1.0


def test_fuse_names_official_objects_from_matching_claims():
    off = [{"masks": {("A", 0): box(0, 10)}, "cost": 0.4},       # matches the bottle claim
           {"masks": {("B", 1): box(12, 18)}, "cost": 0.6}]      # nothing claimed there
    claims = [{"name": "bottle", "change": "removed", "description": "on the sink", "score": 0.9,
               "masks": {("A", 0): box(1, 10), ("A", 2): box(3, 8)}},
              {"name": "towel", "change": "added", "description": "on the rail", "score": 0.7,
               "masks": {("B", 4): box(0, 4)}}]
    agreed, off_only, vlm_only = fuse(off, claims)
    assert [a["name"] for a in agreed] == ["bottle"] and agreed[0]["source"] == "both"
    assert (agreed[0]["masks"][("A", 0)] == box(0, 10)).all()   # the official mask wins on a shared frame
    assert ("A", 2) in agreed[0]["masks"]                        # the VLM fills frames the official method missed
    assert off_only == [off[1]]
    assert [c["name"] for c in vlm_only] == ["towel"]


def test_fuse_merges_official_pieces_of_one_claimed_object():
    off = [{"masks": {("A", 0): box(0, 10)}, "cost": 0.4}, {"masks": {("A", 0): box(0, 8), ("B", 1): box(2, 9)}, "cost": 0.5}]
    claims = [{"name": "bottle", "change": "moved", "description": "", "score": 0.9,
               "masks": {("A", 0): box(1, 10), ("B", 1): box(2, 9)}}]
    agreed, off_only, vlm_only = fuse(off, claims)
    assert len(agreed) == 1 and not off_only and not vlm_only
    assert (agreed[0]["masks"][("A", 0)] == box(0, 10)).all()  # union of the two official pieces
