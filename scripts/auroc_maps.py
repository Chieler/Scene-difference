"""Per-patch AUROC of the cached change-map terms (out/obj/<scene>/cmaps.npz) against GT masks.

    uv run scripts/auroc_maps.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import bench  # noqa: E402


def gt_patches(g, v, i, hw):
    m = np.zeros(hw, bool)
    for o in g[f"video{v}_objects"].values():
        r = o.get(str(i * 30), o.get(i * 30))
        if r is not None:
            m |= np.asarray(Image.fromarray(mu.decode(r) * 255).resize(hw[::-1], Image.BILINEAR)) > 127
    return m


rows = {k: ([], []) for k in ("app", "geo", "comb", "app_rank")}
for d in sorted(Path("out/obj").iterdir()):
    f = d / "cmaps.npz"
    if not f.exists() or "a1" not in np.load(f):
        continue
    z, g = np.load(f), bench.load_gt(d.name)
    per = {}
    for v in (1, 2):
        a, ge, c = z[f"a{v}"], z[f"g{v}"], z[f"c{v}"]
        # rank-normalize appearance within the scene/video: thresholds then transfer across scenes
        ok = ~np.isnan(a)
        ar = np.full_like(a, np.nan); ar[ok] = np.argsort(np.argsort(a[ok])) / ok.sum()
        for i in range(len(a)):
            y = gt_patches(g, v, i, a.shape[1:])
            for k, x in (("app", a[i]), ("geo", np.nan_to_num(ge[i])), ("comb", c[i]), ("app_rank", ar[i])):
                keep = ~np.isnan(x)
                rows[k][0].append(y[keep]); rows[k][1].append(x[keep])
                per.setdefault(k, ([], []))[0].append(y[keep]); per[k][1].append(x[keep])
    s = []
    for k, (y, x) in per.items():
        y, x = np.concatenate(y), np.concatenate(x)
        s.append(f"{k}={roc_auc_score(y, x):.2f}" if 0 < y.sum() < len(y) else f"{k}=na")
    pos = np.concatenate(per["app"][0]).mean()
    print(f"{d.name:40s} pos={pos:.3%} " + " ".join(s))
print("ALL", " ".join(f"{k}={roc_auc_score(np.concatenate(y), np.concatenate(x)):.3f}" for k, (y, x) in rows.items()))
