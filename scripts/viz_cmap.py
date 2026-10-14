"""Contact sheet: each eval frame with its change map and the GT outline (red). Diagnostic only.

    uv run scripts/viz_cmap.py <scene> [pred_dir]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
import bench  # noqa: E402

scene = sys.argv[1]
pred = Path(sys.argv[2]) / scene / "object_masks.pkl" if len(sys.argv) > 2 else None
c = np.load(f"out/obj/{scene}/cmaps.npz")
g = bench.load_gt(scene)
P = pickle.load(open(pred, "rb")) if pred and pred.exists() else {}
tiles = []
for v in (1, 2):
    for i, p in enumerate(bench.extract(scene, v, f"out/frames/{scene}")):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        cm = np.nan_to_num(c[f"c{v}"][i], nan=0)
        heat = Image.fromarray((np.clip(cm, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
        rgb = np.asarray(im).astype(float) * 0.5
        rgb[..., 0] += np.asarray(heat) * 0.5
        out = Image.fromarray(rgb.astype(np.uint8))
        d = ImageDraw.Draw(out)
        gt = np.zeros((H, W), bool)
        for o in g[f"video{v}_objects"].values():
            r = o.get(str(i * 30), o.get(i * 30))
            if r is not None:
                gt |= np.asarray(Image.fromarray(mu.decode(r) * 255).resize((W, H), Image.NEAREST)) > 0
        edge = gt ^ ndimage.binary_erosion(gt, iterations=3)
        a = np.asarray(out).copy(); a[edge] = (0, 255, 0)
        for k, o in P.items():
            if k in ("H", "W"):
                continue
            r = o.get(f"video_{v}", {}).get(i)
            if r is not None:
                m = np.asarray(Image.fromarray(mu.decode(r["mask"]) * 255).resize((W, H), Image.NEAREST)) > 0
                a[m ^ ndimage.binary_erosion(m, iterations=2)] = (0, 128, 255)
        out = Image.fromarray(a)
        ImageDraw.Draw(out).text((8, 8), f"v{v} f{i}", fill=(255, 255, 0))
        tiles.append(out.resize((W // 3, H // 3)))
w, h = tiles[0].size
cols = 8
sheet = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)))
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols) * w, (k // cols) * h))
sheet.save(f"out/obj/{scene}/viz.jpg")
print(f"out/obj/{scene}/viz.jpg")
