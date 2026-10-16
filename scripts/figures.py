"""README figures: benchmark progression and the headline change of the committed example. CPU only.

    uv run scripts/figures.py     # -> docs/benchmark.png, docs/headline.jpg
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# obj/im AP, SceneDiff val, 50 scenes, official scorer (HANDOFF.md tables)
BENCH = [
    ("Our VLM only\n(VLM + SAM2)", 0.322),
    ("Official SceneDiff\n(MPS port) = offline", 0.548),
    ("Fusion", 0.576),
    ("Fusion + VLM verifier\n= full version", 0.585),
]


def benchmark():
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    names, ap = zip(*BENCH)
    colors = ["#b8bcc4"] * (len(ap) - 1) + ["#2a6fdb"]
    bars = ax.barh(names[::-1], ap[::-1], color=colors[::-1], height=0.6)
    for b, v in zip(bars, ap[::-1]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=10, color="#222")
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("obj/im AP (higher is better)", color="#444")
    ax.set_title("SceneDiff val, 50 scenes, official scorer", loc="left", fontsize=11, color="#222")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, colors="#444")
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DOCS / "benchmark.png")


def headline(ex=ROOT / "examples" / "full"):
    c = json.load(open(ex / "changes.json"))[0]
    im = Image.open(ex / c["image"])
    fig, ax = plt.subplots(figsize=(im.width / 70, im.height / 70 + 0.5), dpi=100)
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(f"{c['type'].capitalize()} {c['shift_m']:.2f} m: {c['label']} ({c['status']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "headline.jpg", pil_kwargs={"quality": 85})


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    benchmark()
    headline()
