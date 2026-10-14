#!/usr/bin/env python3
"""VLM object-change detection with per-frame SAM2 masks for SceneDiff."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402

VLM_MODEL = "codex-default"
SAM_MODEL = "facebook/sam2.1-hiera-base-plus"
SHEET_W, SHEET_H = 320, 180
SHEET_COLS, SHEET_ROWS = 3, 4


def frames_for(scene: str, video: str) -> list[tuple[int, Path]]:
    d = ROOT / "out/frames" / scene / ("v1" if video == "video_1" else "v2")
    return [(int(p.stem), p) for p in sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))]


def make_sheets(scene: str, video: str, frames: list[tuple[int, Path]], tmp: Path) -> list[Path]:
    """Create labeled sheets; the VLM reports boxes local to each labeled tile."""
    per_sheet = SHEET_COLS * SHEET_ROWS
    result = []
    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(frames), per_sheet):
        chunk = frames[start : start + per_sheet]
        sheet = Image.new("RGB", (SHEET_COLS * SHEET_W, SHEET_ROWS * (SHEET_H + 24)), "#202124")
        draw = ImageDraw.Draw(sheet)
        for j, (idx, path) in enumerate(chunk):
            col, row = j % SHEET_COLS, j // SHEET_COLS
            x, y = col * SHEET_W, row * (SHEET_H + 24)
            im = Image.open(path).convert("RGB")
            im.thumbnail((SHEET_W, SHEET_H), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (SHEET_W, SHEET_H), "black")
            tile.paste(im, ((SHEET_W - im.width) // 2, (SHEET_H - im.height) // 2))
            sheet.paste(tile, (x, y + 24))
            label = f"{video} frame {idx:05d}"
            draw.text((x + 5, y + 3), label, fill="white", font=font)
        path = tmp / f"{video}_{start // per_sheet:02d}.jpg"
        sheet.save(path, quality=92)
        result.append(path)
    return result


def prompt_for(scene: str, frame_info: dict[str, list[int]]) -> str:
    labels = "\n".join(f"{video}: {', '.join(map(str, ids))}" for video, ids in frame_info.items())
    return f"""Compare the before and after room captures for SceneDiff scene {scene}.
The attached images are labeled contact sheets. Each tile is a full video frame; its label gives its video and exact frame index. Frame indices are the integer filename stem.

Identify only physical objects whose state changed between captures: added (visible only in video_2), removed (visible only in video_1), or moved (the same object at clearly different locations). Ignore camera/viewpoint differences, lighting, shadows, reflections, people, and unchanged objects. Do not call an object moved when only camera angle or occlusion differs. Treat a moved object as ONE identity with boxes in both videos. Boxes should tightly cover the object, excluding nearby background.

For each changed object, provide a bounding box in EVERY attached frame where any part of that object is visible. Use coordinates normalized from 0 to 1000 relative to the full tile (not the contact sheet): [x_min,y_min,x_max,y_max], origin top-left. Include frame indices exactly as labeled. Include an honest confidence from 0 to 1. Return JSON only, with this schema:
{{"objects":[{{"name":"short object name","change":"added|removed|moved","confidence":0.0,"video_1":{{"frame_index":[x1,y1,x2,y2]}},"video_2":{{"frame_index":[x1,y1,x2,y2]}}}}]}}
Use an empty map for the video where an added/removed object is absent. Never include an unchanged object. No markdown fences or other text.

Attached frame indices:
{labels}
"""


def call_vlm(scene: str, frames: dict[str, list[tuple[int, Path]]], cache: Path, force: bool = False) -> dict:
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    import tempfile

    with tempfile.TemporaryDirectory(prefix="scenediff-contact-") as td:
        tmp = Path(td)
        sheets = []
        frame_info = {}
        for video, items in frames.items():
            frame_info[video] = [idx for idx, _ in items]
            sheets.extend(make_sheets(scene, video, items, tmp))
        prompt = prompt_for(scene, frame_info)
        # Put the prompt before repeatable -i options; this CLI parses -i's
        # trailing file list greedily when it appears before the positional prompt.
        command = ["codex", "exec", "--ephemeral", prompt]
        for path in sheets:
            command.extend(["-i", str(path)])
        started = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        record = {
            "scene": scene,
            "model": VLM_MODEL,
            "command": "codex exec --ephemeral '<prompt>' " + shlex.join(command[4:]),
            "prompt": prompt,
            "frame_indices": frame_info,
            "raw_response": run.stdout,
            "stderr": run.stderr,
            "returncode": run.returncode,
            "vlm_seconds": elapsed,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(record, indent=2))
        if run.returncode:
            raise RuntimeError(f"VLM failed for {scene}: {run.stderr[-1500:]}")
        return record


def parse_objects(raw: str) -> list[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise ValueError(f"VLM response did not contain JSON: {text[:500]}")
        decoded = json.loads(text[left : right + 1])
    objects = decoded.get("objects") if isinstance(decoded, dict) else decoded
    if not isinstance(objects, list):
        raise ValueError("Expected a JSON object with an 'objects' list")
    return objects


def normalize_box(box, width: int, height: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    # Model was asked for 0..1000 normalized coordinates; accept pixel coords too.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1000:
        return None
    x1, x2 = sorted((int(round(x1 * width / 1000)), int(round(x2 * width / 1000))))
    y1, y2 = sorted((int(round(y1 * height / 1000)), int(round(y2 * height / 1000))))
    x1, x2 = max(0, min(width - 1, x1)), max(1, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_sam(device_arg: str):
    import torch
    from transformers import Sam2Model, Sam2Processor

    if device_arg == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = device_arg
    print(f"Loading SAM2 {SAM_MODEL} on {device}", flush=True)
    processor = Sam2Processor.from_pretrained(SAM_MODEL)
    model = Sam2Model.from_pretrained(SAM_MODEL).to(device).eval()
    return processor, model, device


def masks_from_boxes(scene: str, parsed: list[dict], frames: dict[str, list[tuple[int, Path]]], processor, model, device: str):
    import torch

    frame_paths = {video: {idx: path for idx, path in items} for video, items in frames.items()}
    image_cache = {}
    grouped = defaultdict(list)
    for obj_i, obj in enumerate(parsed):
        for video in ("video_1", "video_2"):
            frame_boxes = obj.get(video, {})
            if not isinstance(frame_boxes, dict):
                continue
            for frame_key, raw_box in frame_boxes.items():
                try:
                    idx = int(frame_key)
                except (ValueError, TypeError):
                    continue
                path = frame_paths.get(video, {}).get(idx)
                if path is None:
                    continue
                if (video, idx) not in image_cache:
                    image_cache[(video, idx)] = Image.open(path).convert("RGB")
                image = image_cache[(video, idx)]
                box = normalize_box(raw_box, image.width, image.height)
                if box is not None:
                    grouped[(video, idx)].append((obj_i, box))

    object_masks = [{"video_1": {}, "video_2": {}, "score": float(np.clip(obj.get("confidence", 0.5), 0, 1))}
                    for obj in parsed]
    for (video, idx), entries in grouped.items():
        image = image_cache[(video, idx)]
        box_inputs = [[[float(v) for v in box] for _, box in entries]]
        inputs = processor(images=image, input_boxes=box_inputs, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
        )[0]
        ious = outputs.iou_scores.detach().cpu()[0]
        for prompt_i, (obj_i, _) in enumerate(entries):
            best_i = int(torch.argmax(ious[prompt_i]).item())
            mask = masks[prompt_i, best_i].numpy().astype(bool)
            if mask.any():
                object_masks[obj_i][video][idx] = mask
        del inputs, outputs, masks, ious
        if device == "mps":
            torch.mps.empty_cache()
    return object_masks


def scene_available(scene: str) -> bool:
    for v in (1, 2):
        try:
            p = bench.video_path(scene, v)
        except (StopIteration, FileNotFoundError):
            return False
        if not p.is_file() or not frames_for(scene, f"video_{v}"):
            return False
    return True


def process_scene(scene: str, args, sam_state):
    start = time.perf_counter()
    frames = {v: frames_for(scene, v) for v in ("video_1", "video_2")}
    raw_cache = ROOT / "out/vlm" / f"{scene}.json"
    response = call_vlm(scene, frames, raw_cache, args.force_vlm)
    objects = parse_objects(response["raw_response"])
    if sam_state[0] is None:
        sam_state[:] = load_sam(args.device)
    predicted = masks_from_boxes(scene, objects, frames, *sam_state)
    h, w = np.asarray(Image.open(frames["video_1"][0][1])).shape[:2]
    out = ROOT / "out/pred_vlm" / scene / "object_masks.pkl"
    bench.save_pred(out, predicted, h, w)
    return {"scene": scene, "objects": len(objects), "masks": sum(len(o["video_1"]) + len(o["video_2"]) for o in predicted),
            "seconds": time.perf_counter() - start, "vlm_seconds": response.get("vlm_seconds", 0), "prediction": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", help="Process only these scene IDs")
    parser.add_argument("--limit", type=int, help="Process first N available scenes")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--force-vlm", action="store_true", help="Ignore raw-response cache and query VLM again")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    requested = args.scenes or bench.scenes()
    available = [s for s in requested if scene_available(s)]
    missing = [s for s in requested if s not in available]
    for scene in missing:
        print(f"Skipping {scene}: videos or extracted frames missing", flush=True)
    if args.limit:
        available = available[: args.limit]
    if not available:
        raise SystemExit("No scenes with both videos and extracted frames are available")

    sam_state = [None, None, None]
    timings = []
    for i, scene in enumerate(available, 1):
        print(f"[{i}/{len(available)}] {scene}", flush=True)
        try:
            result = process_scene(scene, args, sam_state)
            timings.append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            print(f"FAILED {scene}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise
    timing_path = ROOT / "out/pred_vlm" / "timings.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timings, indent=2))
    if not args.no_score:
        print(bench.official_eval(ROOT / "out/pred_vlm", ROOT / "out/pred_vlm/result.txt"))
    if timings:
        seconds = [r["seconds"] for r in timings]
        print(f"Runtime per scene: mean={np.mean(seconds):.1f}s median={np.median(seconds):.1f}s n={len(seconds)}")


if __name__ == "__main__":
    main()
