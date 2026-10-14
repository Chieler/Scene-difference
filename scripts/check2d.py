"""How good is the raw 2D change map, before any 3D lifting? Score per patch against ground-truth masks."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diff, eval as ev, change2d
from pycocotools import mask as rle

change2d.MAX_DEPTH = np.inf  # keep every patch so scores reshape back to the frame grid
OUT, DATA = ROOT / "out/val", ROOT / "data/scenediff/scenediff_benchmark/data"
THR = 0.7


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


pairs = sorted(d for d in OUT.iterdir() if (d / "cams.npz").exists())[::4][:12]
obj_hit, obj_tot, aucs, bg_fire, fg_fire, unseen = 0, 0, [], [], [], []
for od in pairs:
    z = np.load(od / "recon.npz")
    V = {s: diff.Views(*(z[f"{s}{f}"] for f in diff.FIELDS)) for s in "AB"}
    F = {s: change2d.dino(V[s].images) for s in "AB"}
    S, H, W = V["A"].depth.shape
    _, h, w, _ = F["A"].shape
    gts = ev.load_gt(DATA / od.name)
    for side, other, vid in (("A", "B", 1), ("B", "A", 2)):
        _, s = change2d.scores(V[side], V[other], F[side], F[other])
        s = s.reshape(S, h, w)
        total = ev.frame_count(next((DATA / od.name).glob(f"original_video{vid}.*")))
        gtmask = np.zeros((S, h, w), bool)
        best = {}
        for k in range(S):
            for j, g in enumerate(gts):
                m = g[side] and g[side].get(str(round(k * total / S)))
                if not m or rle.area(m) == 0:
                    continue
                mm = np.asarray(Image.fromarray(rle.decode(m) * 255).resize((w, h), Image.BILINEAR)) > 64
                if not mm.any():
                    continue
                gtmask[k] |= mm
                v = s[k][mm]
                best[j] = max(best.get(j, 0), np.nanmean(np.nan_to_num(v, nan=0) > THR))
        for k in range(S):
            sk, gk = s[k], gtmask[k]
            ok = ~np.isnan(sk)
            unseen.append(1 - ok.mean())
            if gk.any():
                aucs.append(auroc(sk[ok & gk], sk[ok & ~gk]))
            bg_fire.append(np.mean(sk[ok & ~gk] > THR))
            if (ok & gk).any():
                fg_fire.append(np.mean(sk[ok & gk] > THR))
        obj_hit += sum(v >= 0.3 for v in best.values()); obj_tot += len(best)
    print(od.name, file=sys.stderr, flush=True)

print(f"pairs {len(pairs)}")
print(f"patch AUROC change-vs-not, frames with GT: {np.nanmean(aucs):.2f}")
print(f"patches > {THR} inside GT masks: {np.mean(fg_fire):.0%} (thr 0.5: n/a)   outside: {np.nanmean(bg_fire):.1%}")
print(f"patches with no comparable view in other video: {np.mean(unseen):.0%}")
print(f"object-sides with >= 30% of mask firing in some frame: {obj_hit}/{obj_tot}")
