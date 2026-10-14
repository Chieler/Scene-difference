"""Score change2d on cached val reconstructions; cache per-pair scores, then sweep vote thresholds."""
import itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diff, eval as ev, change2d
from changes import Grid

OUT, DATA = ROOT / "out/val", ROOT / "data/scenediff/scenediff_benchmark/data"
CACHE = ROOT / "out/s2d"; CACHE.mkdir(exist_ok=True)
only = sys.argv[1:]

pairs, missed = [], 0
for od in sorted(d for d in OUT.iterdir() if (DATA / d.name / "segments.pkl").exists()):
    if only and od.name not in only:
        continue
    if not (od / "cams.npz").exists():
        missed += len(ev.load_gt(DATA / od.name)); continue
    c = CACHE / f"{od.name}.npz"
    if not c.exists():
        z = np.load(od / "recon.npz")
        A, B = (diff.Views(*(z[f"{s}{f}"] for f in diff.FIELDS)) for s in "AB")
        fa, fb = change2d.dino(A.images), change2d.dino(B.images)
        pr, sr = change2d.scores(A, B, fa, fb)
        pa, sa = change2d.scores(B, A, fb, fa)
        np.savez(c, pr=pr, sr=sr, pa=pa, sa=sa)
        print(f"scored {od.name}", file=sys.stderr, flush=True)
    pairs.append((od, np.load(c)))

grids = {od: Grid.around(z["pr"], z["pa"]) for od, z in pairs}
rows = []
for thr, ratio, mv in itertools.product((0.6, 0.7, 0.75, 0.8, 0.85), (0.3, 0.5, 0.7), (8, 20, 40, 80)):
    tot = dict.fromkeys(("preds", "tp", "gts", "found", "typed"), 0); tot["gts"] = missed
    for od, z in pairs:
        g = grids[od]
        bl = change2d.vote(z["pa"], z["sa"], g, "added", thr, ratio, 2, mv) + \
            change2d.vote(z["pr"], z["sr"], g, "removed", thr, ratio, 2, mv)
        preds = [{"type": b.kind, "lo": b.lo, "hi": b.hi} for b in bl]
        p, gt, sim = ev.eval_pair(od, DATA / od.name, preds)
        s = ev.score(p, gt, sim)
        tot = {k: tot[k] + s[k] for k in tot}
    pr, rc = tot["tp"] / max(1, tot["preds"]), tot["found"] / max(1, tot["gts"])
    rows.append((2 * pr * rc / max(1e-9, pr + rc), pr, rc, tot["typed"] / max(1, tot["gts"]), tot["preds"], (thr, ratio, mv)))
print("   f1  prec  rec  typed  preds   thr ratio min_vox")
for f1, p, r, t, n, c in sorted(rows, reverse=True)[:12]:
    print(f" {f1:.2f}  {p:.2f}  {r:.2f}  {t:.2f}  {n:5d}  {c[0]:4.1f} {c[1]:5.1f} {c[2]:7d}")
print("best recall:", max(rows, key=lambda x: x[2])[:5], max(rows, key=lambda x: x[2])[-1])
