"""Score diff.py output against SceneDiff ground-truth masks.

    uv run eval.py out/scenediff data/scenediff/scenediff_benchmark/data

Each predicted 3D box is projected into the sampled frames (A side into video 1, B side into video 2)
and compared with the 2D box of the ground-truth mask in the same frame.
"""
import argparse, itertools, json, pickle, subprocess
from pathlib import Path
import numpy as np
from pycocotools import mask as rle

MATCH_IOU = 0.2  # a projected 3D box is looser than the mask's 2D box, so the bar is low


def project_rect(lo, hi, K, E, hw):
    """2D rect (x0, y0, x1, y1) of a 3D box in one view, clipped to the image; None if not in view."""
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    cam = corners @ E[:, :3].T + E[:, 3]
    if (cam[:, 2] < 0.1).any():  # ponytail: drops boxes straddling the camera plane, clip in 3D if it matters
        return None
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    H, W = hw
    (x0, y0), (x1, y1) = np.clip(uv.min(0), 0, [W, H]), np.clip(uv.max(0), 0, [W, H])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def side_iou(box, gt_rects, K, E, hw):
    """Mean IoU over the frames where the ground-truth object is visible."""
    vals = [iou(project_rect(*box, K[k], E[k], hw), r) for k, r in enumerate(gt_rects) if r is not None]
    return float(np.mean(vals)) if vals else 0.0


def gt_rects(masks, n, total, hw):
    """Ground-truth 2D box per sampled frame, in our image coordinates. None where not visible."""
    out = []
    for k in range(n):
        m = masks.get(str(round(k * total / n)))  # frame k of ffmpeg's fps=n/dur sampling
        if m is None or rle.area(m) == 0:
            out.append(None)
            continue
        x, y, w, h = rle.toBbox(m)
        sy, sx = hw[0] / m["size"][0], hw[1] / m["size"][1]
        out.append((x * sx, y * sy, (x + w) * sx, (y + h) * sy))
    return out


def frame_count(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip().rstrip(","))


def load_gt(pair_dir):
    s = pickle.load(open(pair_dir / "segments.pkl", "rb"))
    kind = {(True, False): "removed", (False, True): "added", (True, True): "moved"}
    return [{"label": o["label"], "type": kind[o["in_video1"], o["in_video2"]],
             "A": s["video1_objects"].get(o["original_obj_idx"]) if o["in_video1"] else None,
             "B": s["video2_objects"].get(o["original_obj_idx"]) if o["in_video2"] else None} for o in s["objects"]]


def pred_sides(c):
    if c["type"] == "moved":
        return {"A": (c["from_lo"], c["from_hi"]), "B": (c["lo"], c["hi"])}
    return {"A" if c["type"] == "removed" else "B": (c["lo"], c["hi"])}


def score(preds, gts, sim):
    """sim[i, j]: overlap of prediction i with ground-truth object j. A prediction may hit several objects."""
    hit = sim >= MATCH_IOU
    return {"preds": len(preds), "tp": int(hit.any(1).sum()), "gts": len(gts), "found": int(hit.any(0).sum()),
            "typed": sum(any(hit[i, j] and p["type"] == g["type"] for i, p in enumerate(preds))
                         for j, g in enumerate(gts))}


def eval_pair(out_dir, pair_dir, preds=None):
    preds = json.loads((out_dir / "changes.json").read_text()) if preds is None else preds
    cams = np.load(out_dir / "cams.npz")
    hw = tuple(cams["hw"])
    gts = load_gt(pair_dir)
    rects = {}
    for side, v in (("A", 1), ("B", 2)):
        n, total = len(cams[f"E{side}"]), frame_count(next(pair_dir.glob(f"original_video{v}.*")))
        for j, g in enumerate(gts):
            if g[side] is not None:
                rects[j, side] = gt_rects(g[side], n, total, hw)
    sim = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for side, box in pred_sides(p).items():
            for j in range(len(gts)):
                if (j, side) in rects:
                    sim[i, j] = max(sim[i, j], side_iou(box, rects[j, side], cams[f"K{side}"], cams[f"E{side}"], hw))
    return preds, gts, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="dir with one diff.py output dir per pair")
    ap.add_argument("data", type=Path, help="SceneDiff data dir")
    a = ap.parse_args()
    tot = dict.fromkeys(("preds", "tp", "gts", "found", "typed"), 0)
    for od in sorted(d for d in a.out.iterdir() if (a.data / d.name / "segments.pkl").exists()):
        if not (od / "cams.npz").exists():  # diff.py failed: every real change is missed
            n = len(load_gt(a.data / od.name))
            tot["gts"] += n
            print(f"== {od.name}: pipeline failed, 0/{n} objects found")
            continue
        preds, gts, sim = eval_pair(od, a.data / od.name)
        s = score(preds, gts, sim)
        tot = {k: tot[k] + s[k] for k in tot}
        print(f"== {od.name}: {s['tp']}/{s['preds']} predictions hit, {s['found']}/{s['gts']} objects found, "
              f"{s['typed']} with right type")
        for j, g in enumerate(gts):
            i = int(sim[:, j].argmax()) if len(preds) else None
            best = f"{sim[i, j]:.2f} {preds[i]['type']} {preds[i]['label']}" if i is not None else "-"
            print(f"   {g['type']:8} {g['label']:28} best: {best}")
    p, r = tot["tp"] / max(1, tot["preds"]), tot["found"] / max(1, tot["gts"])
    print(f"\nprecision {p:.2f}  recall {r:.2f}  type-correct {tot['typed']}/{tot['gts']}  (IoU >= {MATCH_IOU})")


if __name__ == "__main__":
    main()
