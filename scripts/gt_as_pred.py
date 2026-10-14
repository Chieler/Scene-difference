"""Sanity check for the scorer: write GT as predictions, expect every metric at 1.0.

    uv run scripts/gt_as_pred.py out/gt_pred
"""
import sys
from pathlib import Path

from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent.parent))
import bench  # noqa: E402

out = Path(sys.argv[1])
for s in bench.scenes():
    if not (bench.DATA / "data" / s / "segments.pkl").exists():
        continue
    g = bench.load_gt(s)
    objs = []
    for m in g["objects"]:
        k, o = m["original_obj_idx"], {"score": 1.0}
        for vk, gk, flag in (("video_1", "video1_objects", "in_video1"), ("video_2", "video2_objects", "in_video2")):
            if m.get(flag) and k in g[gk]:
                o[vk] = {int(f) // bench.RATE: mask_utils.decode(r) for f, r in g[gk][k].items() if int(f) % bench.RATE == 0}
        objs.append(o)
    H, W = next((r["size"] for gk in ("video1_objects", "video2_objects") for fr in g[gk].values() for r in fr.values()), (1920, 1080))
    bench.save_pred(out / s / "object_masks.pkl", objs, H, W)
print(bench.official_eval(out, out / "result.txt"))
