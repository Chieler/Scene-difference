"""The official SceneDiff method (MPS port) as a candidate source for diff.py.

    candidates(pa, pb, work) -> [{"A": {i: mask}, "B": {i: mask}, "cost": float}, ...]

The port runs on exactly the frames diff.py sampled, in a subprocess so its Pi3 / SAM-H / DINOv2
memory is gone before diff.py continues. Its object_masks.pkl is cached in `work`.
Needs `bash scripts/setup_scenediff.sh` once (Pi3, SAM, weights); after that it runs offline.
"""
import json, os, pickle, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

PORT = Path(__file__).resolve().parent / "scenediff_mps"
CONFIG = PORT / "configs" / "scenediff_mps.yml"
SIDES = {"video_1": "A", "video_2": "B"}


def _key(pa, pb):
    """Frame contents, so a copied or moved output folder keeps its cache."""
    import hashlib
    return [len(pa), len(pb)] + [hashlib.sha1(p.read_bytes()).hexdigest() for p in [*pa, *pb]]


def candidates(pa, pb, work):
    """One dict per object the port found: frame index into pa / pb -> bool mask at that jpg's
    resolution, and 'cost' = its highest per-frame cost (the port's confidence; higher ranks first)."""
    pa, pb, work = [Path(p) for p in pa], [Path(p) for p in pb], Path(work).resolve()  # child runs in PORT
    cache, key = work / "official.pkl", _key(pa, pb)
    raw = None
    if cache.exists():
        c = pickle.load(open(cache, "rb"))
        raw = c["masks"] if c["key"] == key else None
    if raw is None:
        raw = _run(pa, pb, work / "official")
        pickle.dump({"key": key, "masks": raw}, open(cache, "wb"))
    return _convert(raw, pa, pb)


def _run(pa, pb, root):
    """Lay the frames out the way the port expects (<scene>/video{1,2}_frames, SAM cache next to them)
    and run it. A fresh dir each time: the port reuses any sam_masks.pkl it finds."""
    missing = [p for p in ("submodules/Pi3/.mps_patched", "submodules/segment-anything-langsplat-modified/.mps_patched",
                           "checkpoints/sam_vit_h_4b8939.pth") if not (PORT / p).exists()]
    if missing:
        raise RuntimeError(f"official SceneDiff not set up ({', '.join(missing)}): run bash scripts/setup_scenediff.sh")
    shutil.rmtree(root, ignore_errors=True)
    scene = root / "scene"
    files = []
    for name, ps in (("video1_frames", pa), ("video2_frames", pb)):
        (scene / name).mkdir(parents=True)
        for i, p in enumerate(ps):
            files.append(scene / name / f"{i:05d}.jpg")
            files[-1].symlink_to(p.resolve())
    job = root / "job.json"
    job.write_text(json.dumps({"files": [str(f) for f in files], "n1": len(pa), "n2": len(pb),
                               "out": str(root / "out")}))
    log = root / "log.txt"
    env = {"HF_HUB_OFFLINE": "1", **os.environ,  # weights come from the setup script
           "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(job)],
                           cwd=PORT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = "".join(open(log).readlines()[-20:])
        raise RuntimeError(f"official SceneDiff failed (exit {r.returncode}), log {log}:\n{tail}")
    return pickle.load(open(root / "out" / "scene" / "object_masks.pkl", "rb"))


def _convert(raw, pa, pb):
    from pycocotools import mask as mask_utils
    sizes = {"A": Image.open(pa[0]).size, "B": Image.open(pb[0]).size}  # (W, H)
    out = []
    for k, o in raw.items():
        if k in ("H", "W"):
            continue
        e, costs = {"A": {}, "B": {}}, []
        for vk, frames in o.items():
            s = SIDES[vk]
            for f, d in frames.items():
                m = mask_utils.decode(d["mask"]).astype(bool)
                if m.shape[::-1] != sizes[s]:
                    m = np.asarray(Image.fromarray(m).resize(sizes[s], Image.NEAREST))
                e[s][int(f)] = m
                costs.append(d["cost"])
        out.append({**e, "cost": float(max(costs))})
    return out


def _child(job):
    """Subprocess side: the port's predict_multiview.py for one scene, cwd = the port."""
    job = json.loads(Path(job).read_text())
    sys.path[:0] = [str(PORT), str(PORT / "scripts")]
    import torch
    import predict_multiview as pm
    from modules import SceneDiff
    pm.set_random_seed(42)
    config = pm.load_config(CONFIG)
    config["output"]["output_dir"] = job["out"]
    config["visualization"]["save_gt_visualization"] = False
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    SceneDiff(config, device).process_scene(job["files"], job["n1"], job["n2"], job["out"])


if __name__ == "__main__":
    _child(sys.argv[1])
