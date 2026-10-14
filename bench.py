"""SceneDiff benchmark plumbing: scene lists, eval-frame extraction, object_masks.pkl writer, official scorer.

GT is annotated on every 30th frame of the 30 fps video, so a "frame index" here is video_frame // 30,
the same index evaluate_multiview.py uses.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).parent
DATA = ROOT / "data/scenediff/scenediff_benchmark"
OFFICIAL = ROOT / "third_party/scene_diff"
RATE = 30
MAX_SIDE = 1024  # the official scorer resizes everything to this long side


def scenes(split="val", subset="varied"):
    return json.load(open(OFFICIAL / f"splits/{split}_split.json"))[subset]


def video_path(scene, v):
    return next((DATA / "data" / scene).glob(f"original_video{v}.*"))


def extract(scene, v, out):
    """Every 30th frame (official: ffmpeg -r 30, then [::30]), long side MAX_SIDE. Returns sorted paths."""
    d = Path(out) / f"v{v}"
    if not (d.exists() and any(d.iterdir())):
        d.mkdir(parents=True, exist_ok=True)
        vf = f"fps=30,select='not(mod(n\\,{RATE}))',scale='if(gt(iw,ih),{MAX_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_SIDE})'"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video_path(scene, v)), "-vf", vf,
                        "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0", str(d / "%05d.jpg")], check=True)
    return sorted(d.glob("*.jpg"))


def load_frames(paths):
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def rle(mask):
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def save_pred(path, objects, H, W):
    """objects: list of {'video_1': {frame: bool mask}, 'video_2': {...}, 'score': float}.

    Layout follows evaluate_multiview.extract_detections (not the README): {obj_id: {'video_k': {frame:
    {'mask', 'cost'}}}, 'H', 'W'}. An object present in both videos is scored as "moved".
    'cost' is a confidence: higher ranks first.
    """
    out = {"H": H, "W": W}
    for i, o in enumerate(objects):
        e = {}
        for vk in ("video_1", "video_2"):
            fr = {int(f): {"mask": rle(m), "cost": float(o.get("frame_scores", {}).get((vk, f), o["score"]))}
                  for f, m in o.get(vk, {}).items() if m.any()}
            if fr:
                e[vk] = fr
        if e:
            out[i] = e
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)


def load_gt(scene):
    with open(DATA / "data" / scene / "segments.pkl", "rb") as f:
        return pickle.load(f)


def official_eval(pred_dir, out_txt, split="val", subset="varied"):
    """Run third_party/scene_diff/scripts/evaluate_multiview.py; returns the metric lines it prints."""
    pred_dir, out_txt = Path(pred_dir).resolve(), Path(out_txt).resolve()
    cmd = [sys.executable, str(OFFICIAL / "scripts/evaluate_multiview.py"), "--pred_dir", str(pred_dir),
           "--gt_dir", str(DATA / "data"), "--video_dir", str(DATA / "data"), "--output_path", str(out_txt),
           "--splits", split, "--sets", subset]
    # the script reads data/scenediff_benchmark/splits relative to cwd
    cwd = ROOT / "out/evalcwd"
    (cwd / "data/scenediff_benchmark").mkdir(parents=True, exist_ok=True)
    if not (cwd / "data/scenediff_benchmark/splits").exists():
        (cwd / "data/scenediff_benchmark/splits").symlink_to(OFFICIAL / "splits")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return "\n".join(l for l in r.stdout.splitlines() if l.startswith(("Evaluating", "Not predicted", "Metric", "  T", "  F")))


if __name__ == "__main__":
    print(official_eval(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]) / "result.txt"))
