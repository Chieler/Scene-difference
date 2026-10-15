"""2-fold cross-validated choice among the fused / verified variants, so no scene is scored by a variant picked on it.

The 50 val/varied scenes are split in two halves (seeded shuffle, as scripts/cv_retype.py). For each half the
variant with the best obj/im AP on the other half is used; the two held-out halves are then scored together as one
50-scene prediction.

    uv run scripts/cv_select.py        # needs the pred dirs below (scripts/fuse.py, scripts/verify.py)
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cv_retype import metrics  # noqa: E402

VARIANTS = ["pred_official", "pred_fuse_agree", "pred_fuse_union_off", "pred_fuse_union_vlm",
            "pred_verify_drop", "pred_verify_demote", "pred_verify_conf"]
SETS = ROOT / "third_party/scene_diff/splits"


def score(pred, name, scenes, cwd):
    """Official evaluator on `scenes`: the evaluator only takes its own set names, so the patched val_split.json
    in cwd lists them as 'varied'."""
    json.dump({"varied": scenes, "kitchen": []}, open(cwd / "data/scenediff_benchmark/splits/val_split.json", "w"))
    r = subprocess.run([sys.executable, str(ROOT / "third_party/scene_diff/scripts/evaluate_multiview.py"),
                        "--pred_dir", str(pred), "--gt_dir", str(ROOT / "data/scenediff/scenediff_benchmark/data"),
                        "--video_dir", str(ROOT / "data/scenediff/scenediff_benchmark/data"),
                        "--output_path", str(cwd / f"{pred.name}_{name}.txt"), "--splits", "val", "--sets", "varied"],
                       cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return metrics(r.stdout)


if __name__ == "__main__":
    cwd = ROOT / "out/cv_select"
    splits = cwd / "data/scenediff_benchmark/splits"
    splits.mkdir(parents=True, exist_ok=True)
    val = json.load(open(SETS / "val_split.json"))
    scenes = sorted(val["varied"])
    shuffled = scenes[:]
    random.Random(0).shuffle(shuffled)
    folds = {"fold0": sorted(shuffled[::2]), "fold1": sorted(shuffled[1::2])}
    shutil.copy(SETS / "test_split.json", splits / "test_split.json")

    ap = {(v, f): score(ROOT / "out" / v, f, folds[f], cwd)["obj_im_ap"] for v in VARIANTS for f in folds}
    for v in VARIANTS:
        print(f"{v:22s} fold0 {ap[v, 'fold0']:.3f}  fold1 {ap[v, 'fold1']:.3f}")
    out = cwd / "pred_cv"
    shutil.rmtree(out, ignore_errors=True)
    for test, train in (("fold0", "fold1"), ("fold1", "fold0")):
        best = max(VARIANTS, key=lambda v: ap[v, train])
        print(f"{test}: picked {best} on {train}")
        for s in folds[test]:
            shutil.copytree(ROOT / "out" / best / s, out / s)
    print("held-out, 50 scenes:", score(out, "all", scenes, cwd))
