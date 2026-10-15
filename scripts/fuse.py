"""Fuse the official SceneDiff predictions (high recall, many false alarms) with the VLM track (precise, misses
small things). Objects both methods found rank first; the rest follow by their own confidence.

    uv run scripts/fuse.py [variant ...]      # see VARIANTS (default: all)

Writes out/pred_fuse_<variant>/<scene>/object_masks.pkl and scores each with the official evaluator.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402

MATCH_IOU = 0.3  # mean per-frame mask IoU over shared frames for "both methods found it"


def load(path, H=None, W=None):
    """Objects as {video: {frame: bool mask}, 'cost': max cost}, masks resized to (H, W)."""
    p = pickle.load(open(path, "rb"))
    out = []
    for k, o in p.items():
        if k in ("H", "W"):
            continue
        e, costs = {}, []
        for vk, frames in o.items():
            e[vk] = {}
            for f, d in frames.items():
                m = mask_utils.decode(d["mask"]).astype(bool)
                if H and m.shape != (H, W):
                    m = np.asarray(Image.fromarray(m).resize((W, H), Image.NEAREST))
                e[vk][int(f)] = m
                costs.append(d["cost"])
        out.append({"masks": e, "cost": max(costs)})
    return out, p["H"], p["W"]


def iou(a, b):
    ious = [(a[vk][f] & b[vk][f]).sum() / max(1, (a[vk][f] | b[vk][f]).sum())
            for vk in a if vk in b for f in a[vk] if f in b[vk]]
    return float(np.mean(ious)) if ious else 0.0


# rank band per group (higher ranks first); None drops the group
VARIANTS = {"agree": (2, 1, None),       # official, reranked: objects the VLM also found first
            "union_off": (2, 1, 0),      # + VLM-only objects after the official-only ones
            "union_vlm": (2, 0, 1)}      # + VLM-only objects before the official-only ones


def fuse(off, vlm, variant):
    """Each object scores band + own confidence in [0, 1), bands (agreed, official-only, VLM-only)."""
    agreed, off_only, used = [], [], set()
    for o in off:
        j, best = max(((j, iou(o["masks"], v["masks"])) for j, v in enumerate(vlm)), key=lambda t: t[1],
                      default=(None, 0.0))
        if best >= MATCH_IOU:
            used.add(j)
            v = vlm[j]  # official masks win where both have a frame (they score higher); the VLM fills the rest
            agreed.append(({vk: {**v["masks"].get(vk, {}), **o["masks"].get(vk, {})}
                            for vk in set(o["masks"]) | set(v["masks"])}, v["cost"]))
        else:
            off_only.append((o["masks"], o["cost"]))
    vlm_only = [(v["masks"], v["cost"]) for j, v in enumerate(vlm) if j not in used]
    objs = []
    for band, group in zip(VARIANTS[variant], (agreed, off_only, vlm_only)):
        if band is not None:
            objs += [{**m, "score": band + min(c, 0.999)} for m, c in group]
    return objs


if __name__ == "__main__":
    variants = sys.argv[1:] or list(VARIANTS)
    scenes = sorted(p.parent.name for p in (ROOT / "out/pred_official").glob("*/object_masks.pkl"))
    for variant in variants:
        out = ROOT / f"out/pred_fuse_{variant}"
        for s in scenes:
            vlm, H, W = load(ROOT / "out/pred_vlm" / s / "object_masks.pkl")
            off, _, _ = load(ROOT / "out/pred_official" / s / "object_masks.pkl", H, W)
            bench.save_pred(out / s / "object_masks.pkl", fuse(off, vlm, variant), H, W)
        print(variant, "\n", bench.official_eval(out, out / "result.txt"), flush=True)
