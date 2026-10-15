"""The VLM as a verifier: every object only the official SceneDiff method found becomes a numbered, outlined crop
(its largest mask per video). The VLM sees those crops plus both videos' contact sheets and says which are real
changes. Rejected candidates drop to the bottom of the ranking.

    uv run scripts/verify.py [scenes...]      # raw answers cached in out/verify/<scene>.json

Writes out/pred_verify_<variant>/<scene>/object_masks.pkl and scores each with the official evaluator.
"""
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bench  # noqa: E402
from fuse import MATCH_IOU, iou, load  # noqa: E402
from vlm_track import frames_for, make_sheets, parse_objects  # noqa: E402

TILE, COLS, ROWS = 320, 4, 3
CONTEXT = 2.5  # crop side = CONTEXT x the mask's larger side, at least a quarter of the frame
WORKERS = 4    # parallel codex calls; no GPU involved

PROMPT = """You are checking candidates from an automatic change detector for SceneDiff scene {scene}.
video_1 is the before capture, video_2 the after capture. The first images are contact sheets of both videos (full
frames, labeled). The remaining images are candidate crops: each tile is labeled "#<id> <video> f<frame>", and the
candidate region is outlined in magenta. A candidate seen only in video_1 was detected as removed, only in video_2
as added, in both as moved.

The detector produces many false alarms: unchanged objects seen from a new angle, parts of walls, floors, furniture
or shadows, regions only partly in view, and people. For each candidate id, decide if the outlined region is a
physical object that really was added, removed or moved between the two captures. Use the contact sheets to check
whether the object is also present, in the same place, in the other video.

Return JSON only, no markdown: {{"candidates":[{{"id":0,"real":true,"name":"short object name","confidence":0.0}}]}}
with one entry for every id: {ids}. confidence is how sure you are that it is a real change (0 to 1)."""


def tile(frame, mask, label, font):
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    side = int(max(CONTEXT * max(ys.ptp(), xs.ptp()), min(H, W) / 2))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0, x0 = np.clip(cy - side // 2, 0, max(0, H - side)), np.clip(cx - side // 2, 0, max(0, W - side))
    im = np.asarray(Image.open(frame).convert("RGB")).copy()
    edge = mask & ~ndimage.binary_erosion(mask, iterations=3)
    im[edge] = (255, 0, 255)
    crop = Image.fromarray(im[y0:y0 + side, x0:x0 + side]).resize((TILE, TILE - 22), Image.LANCZOS)
    t = Image.new("RGB", (TILE, TILE), "#202124")
    t.paste(crop, (0, 22))
    ImageDraw.Draw(t).text((5, 3), label, fill="white", font=font)
    return t


def candidate_sheets(scene, cands, tmp):
    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    paths = {vk: dict(frames_for(scene, vk)) for vk in ("video_1", "video_2")}
    tiles = []
    for i, masks in cands:
        for vk, fr in sorted(masks.items()):
            f = max(fr, key=lambda f: fr[f].sum())  # its largest view in this video
            tiles.append(tile(paths[vk][f], fr[f], f"#{i} {vk} f{f}", font))
    out = []
    for s in range(0, len(tiles), COLS * ROWS):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), "#202124")
        for j, t in enumerate(tiles[s:s + COLS * ROWS]):
            sheet.paste(t, ((j % COLS) * TILE, (j // COLS) * TILE))
        out.append(tmp / f"cand_{s // (COLS * ROWS):02d}.jpg")
        sheet.save(out[-1], quality=92)
    return out


def ask(scene, cands):
    cache = ROOT / "out/verify" / f"{scene}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with tempfile.TemporaryDirectory(prefix="scenediff-verify-") as td:
        tmp = Path(td)
        sheets = [p for vk in ("video_1", "video_2") for p in make_sheets(scene, vk, frames_for(scene, vk), tmp)]
        sheets += candidate_sheets(scene, cands, tmp)
        prompt = PROMPT.format(scene=scene, ids=", ".join(str(i) for i, _ in cands))
        cmd = ["codex", "exec", "--ephemeral", prompt]  # prompt before the -i list (see vlm_track.call_vlm)
        for p in sheets:
            cmd += ["-i", str(p)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{scene}: {r.stderr[-1500:]}")
    ans = {int(c["id"]): c for c in parse_candidates(r.stdout)}
    rec = {"scene": scene, "ids": [i for i, _ in cands], "answers": {str(k): v for k, v in ans.items()}, "raw": r.stdout}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, indent=1))
    return rec


def parse_candidates(raw):
    text = raw.strip()
    l, r = text.find("{"), text.rfind("}")
    return json.loads(text[l:r + 1])["candidates"]


def split(scene):
    """(agreed, official-only, VLM-only), each a list of (masks, confidence); as in fuse.fuse."""
    vlm, H, W = load(ROOT / "out/pred_vlm" / scene / "object_masks.pkl")
    off, _, _ = load(ROOT / "out/pred_official" / scene / "object_masks.pkl", H, W)
    agreed, off_only, used = [], [], set()
    for o in off:
        j, best = max(((j, iou(o["masks"], v["masks"])) for j, v in enumerate(vlm)), key=lambda t: t[1],
                      default=(None, 0.0))
        if best >= MATCH_IOU:
            used.add(j)
            v = vlm[j]
            agreed.append(({vk: {**v["masks"].get(vk, {}), **o["masks"].get(vk, {})}
                            for vk in set(o["masks"]) | set(v["masks"])}, v["cost"]))
        else:
            off_only.append((o["masks"], o["cost"]))
    return agreed, off_only, [(v["masks"], v["cost"]) for j, v in enumerate(vlm) if j not in used], (H, W)


def rescore(agreed, off_only, vlm_only, ans, variant):
    """Bands as in fuse.VARIANTS['union_off']; official-only objects the VLM rejects go below everything
    ('demote') or are dropped ('drop'); 'conf' ranks the official-only band by the VLM's confidence instead."""
    objs = [{**m, "score": 2 + min(c, 0.999)} for m, c in agreed]
    objs += [{**m, "score": min(c, 0.999)} for m, c in vlm_only]
    for i, (m, c) in enumerate(off_only):
        a = ans.get(str(i), {"real": True, "confidence": 0.5})
        if variant == "conf":
            c = float(a.get("confidence", 0.5))  # the model reports confidence in its verdict, not P(real)
            objs.append({**m, "score": 1 + min(c if a.get("real") else 1 - c, 0.999)})
        elif a.get("real"):
            objs.append({**m, "score": 1 + min(c, 0.999)})
        elif variant == "demote":
            objs.append({**m, "score": -1 + min(c, 0.999)})
    return objs


if __name__ == "__main__":
    scenes = sys.argv[1:] or sorted(p.parent.name for p in (ROOT / "out/pred_official").glob("*/object_masks.pkl"))
    parts = {s: split(s) for s in scenes}

    def run(s):
        cands = list(enumerate(m for m, _ in parts[s][1]))
        if not cands:
            return s, {}
        rec = ask(s, cands)
        print(s, len(cands), "candidates,", sum(bool(a.get("real")) for a in rec["answers"].values()), "kept", flush=True)
        return s, rec["answers"]

    with ThreadPoolExecutor(WORKERS) as ex:
        answers = dict(ex.map(run, scenes))
    for variant in ("drop", "demote", "conf"):
        out = ROOT / f"out/pred_verify_{variant}"
        for s in scenes:
            agreed, off_only, vlm_only, (H, W) = parts[s]
            bench.save_pred(out / s / "object_masks.pkl", rescore(agreed, off_only, vlm_only, answers[s], variant), H, W)
        print(variant, "\n", bench.official_eval(out, out / "result.txt"), flush=True)
