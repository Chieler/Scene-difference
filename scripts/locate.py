"""Place every predicted object in 3D: VGGT on the masked frames of both videos (one shared frame, gravity-up,
meters from the floor fit), mask pixels lifted through depth, median per video.

    uv run scripts/locate.py out/pred_vlm [scenes...]

Writes <pred_dir>/<scene>/objects.json: [{id, change, frames: {video_k: [...]}, position: {video_k: [x, y, z]}}].
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recon  # noqa: E402

PER_VIDEO = 6  # VGGT frames per video; masked frames first, evenly subsampled


def pick(masked, n_all):
    fr = sorted(masked)
    if len(fr) > PER_VIDEO:
        fr = [fr[i] for i in np.linspace(0, len(fr) - 1, PER_VIDEO).round().astype(int)]
    rest = [f for f in np.linspace(0, n_all - 1, PER_VIDEO).round().astype(int) if f not in fr]
    return sorted(set(fr) | set(rest[:max(0, 4 - len(fr))]))  # a few context frames when masks are few


def lift(mask, v, i):
    H, W = v.depth.shape[1:]
    m = ndimage.zoom(mask.astype(np.uint8), (H / mask.shape[0], W / mask.shape[1]), order=0).astype(bool)
    m = ndimage.binary_erosion(m, iterations=1) if m.sum() > 50 else m  # drop edge pixels that straddle depth jumps
    if not m.any():
        return np.zeros((0, 3))
    return recon.unproject(v.depth[i], v.K[i], v.E[i])[m]


def locate_scene(d):
    pred = pickle.load(open(d / "object_masks.pkl", "rb"))
    objs = {k: o for k, o in pred.items() if k not in ("H", "W")}
    frames = {vk: sorted((ROOT / "out/frames" / d.name / ("v1" if vk == "video_1" else "v2")).glob("*.jpg"),
                         key=lambda p: int(p.stem)) for vk in ("video_1", "video_2")}
    used = {vk: pick({f for o in objs.values() for f in o.get(vk, {})}, len(frames[vk])) for vk in frames}
    paths = [frames[vk][f] for vk in frames for f in used[vk]]
    imgs = recon.load_frames(paths, recon.frame_size(paths[0]))
    v = recon.Views(imgs, *recon.run_vggt(imgs))
    try:
        s, R, p0 = recon.floor_transform(recon.cloud(v)[0], recon.centers(v), -v.E[:, 1, :3])
        v = recon.align(v, s, R, p0)
    except Exception as e:  # no floor in view: keep VGGT's arbitrary frame
        print(f"{d.name}: no floor fit ({e}); positions in VGGT units", file=sys.stderr)
    n1 = len(used["video_1"])
    views = {"video_1": v.part(0, n1), "video_2": v.part(n1, len(imgs))}
    out = []
    for k, o in objs.items():
        e = {"id": int(k), "change": "moved" if len(o) == 2 else ("removed" if "video_1" in o else "added"),
             "frames": {vk: sorted(int(f) for f in o[vk]) for vk in o}, "position": {}}
        for vk in o:
            pts = [lift(mask_utils.decode(o[vk][f]["mask"]), views[vk], used[vk].index(f))
                   for f in o[vk] if f in used[vk]]
            pts = np.concatenate(pts) if pts else np.zeros((0, 3))
            if len(pts):
                e["position"][vk] = [round(float(x), 3) for x in np.median(pts, 0)]
        out.append(e)
    json.dump(out, open(d / "objects.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    pred_dir = Path(sys.argv[1])
    names = sys.argv[2:] or sorted(p.parent.name for p in pred_dir.glob("*/object_masks.pkl"))
    for n in names:
        if (pred_dir / n / "objects.json").exists():
            continue
        objs = locate_scene(pred_dir / n)
        print(n, len(objs), [o["position"] for o in objs][:3], flush=True)
