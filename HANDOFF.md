# Handoff (2026-10-15)

## Goal

Given a before video and an after video of a room, find objects that were added, removed or moved. For each one, output a per-frame object mask and a 3D position. The benchmark is [SceneDiff](https://github.com/yuqunw/scene_diff) ([paper](https://arxiv.org/abs/2512.16908)).

## Current best: official method + VLM track, fused and verified (`scripts/fuse.py`, `scripts/verify.py`)

Same scorer and scenes as below. Settings were picked on these scenes; confirm on the test split.

| Method | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP + type | TP / FP / FN |
|---|---|---|---|---|---|
| Official SceneDiff, MPS port (`out/pred_official`) | 0.354 | 0.548 | 0.400 | 0.318 | 94 / 319 / 67 |
| Fusion, `union_off` (`out/pred_fuse_union_off`) | 0.490 | 0.576 | 0.498 | 0.448 | 110 / 348 / 51 |
| **+ verifier, demote** (`out/pred_verify_demote`) | 0.490 | **0.585** | **0.522** | **0.472** | 110 / 348 / 51 |
| + verifier, drop (`out/pred_verify_drop`) | **0.605** | 0.548 | 0.494 | 0.449 | 100 / 91 / 61 |

- Fusion: an official object matches a VLM object at mean per-frame mask IoU >= 0.3. Ranking bands: matched, then official-only, then VLM-only. Matched objects keep the official mask where both have a frame (VLM masks there scored 0.503 obj/im); VLM masks fill the other frames.
- Verifier: each official-only object becomes an outlined crop (largest mask per video) on numbered sheets, sent to `codex exec` with both videos' contact sheets. Answers cached in `out/verify/<scene>.json`. It kept 64 of 331. The model's confidence is in its verdict, not P(real); ranking by it (0.544) was worse than the official cost.
- Next: 2-fold CV or the test split; higher-res VLM pass and SAM2 propagation for the 51 misses.

## VLM proposals + SAM2 masks (`scripts/vlm_track.py`)

Official SceneDiff scorer, val split, "varied" set, all 50 scenes.

| Method | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP + type | TP / FP / FN |
|---|---|---|---|---|---|
| **VLM + SAM2 (this repo)** | **0.520** | **0.322** | **0.323** | **0.284** | 69 / 45 / 92 |
| Object-level VGGT + DINOv2 (`objdiff.py`), 1 scene only | 0.008 | 0.024 | 0.125 | | 92 predictions for 2 GT |
| Paper, SceneDiff (π³ + DINOv3), SD-V test | 0.391 | 0.438 | 0.228 | | |
| Paper, VLM baseline (Qwen2.5-VL + SAM3), SD-V test | 0.229 | 0.071 | 0.053 | | |

The paper rows use the test split, not val, so they are reference points only.

How it works:
1. `bench.extract` takes every 30th frame (the GT frames), long side 1024.
2. The frames are tiled into labeled 3x4 contact sheets. `codex exec` (the codex CLI's default model) gets both videos' sheets and one prompt. It returns JSON: each changed object with its change type, a confidence, and a box in every frame where it is visible. Raw responses are cached in `out/vlm/<scene>.json`.
3. SAM2.1 hiera-base-plus (ungated) turns each box into a mask. The mask's `cost` is the VLM confidence.
4. `scripts/locate.py` places each object in 3D:
   - VGGT runs on up to 6 masked frames per video, with both videos in one pass, so they share one frame.
   - Gravity-up and meters come from the floor fit when a floor is found. Otherwise positions are in VGGT units.
   - Each object's position is the median of its eroded mask pixels lifted through depth, per video.
   - Output: `<pred_dir>/<scene>/objects.json` = `[{id, change, frames, position: {video_k: [x, y, z]}}]`.

Run it, about 110 s per scene, of which about 75 s is the VLM call:
```
uv run scripts/vlm_track.py                  # all val/varied scenes -> out/pred_vlm/<scene>/object_masks.pkl
uv run scripts/locate.py out/pred_vlm        # 3D positions -> objects.json
uv run python bench.py out/pred_vlm          # official scorer
```

Where it loses points:
- Misses dominate: 92 FN against 45 FP. The VLM sees 320x180 tiles, and the median missed object covers 0.9% of the frame.
- Next things to try:
  - Higher-resolution sheets, or a second zoomed pass on regions the geometry diff flags.
  - SAM2 video propagation, to fill frames where the VLM gave no box.
  - SAM 3 text prompts, if the gated license is accepted.

## Why the geometry pipelines hit a ceiling

- Earlier `diff.py`, a 2D patch diff voted into voxels: homemade F1 0.27. It produces 3D boxes, not masks, so the official scorer doesn't apply.
- `objdiff.py` reworks it object-first:
  - per-patch DINOv2 change plus a free-space term;
  - SAM2 prompted on change peaks, lifted to 3D, merged across frames.
  - The change map fires nearly everywhere, from the free-space term and textureless cloth, giving 92 predictions for 2 GT objects.
- The root cause is unchanged: VGGT depth disagrees between the two videos by 10 to 15 cm locally, which is as large as the objects. The paper reports the same: VGGT in place of π³ costs 39.8 obj/im AP.
- `scripts/auroc_maps.py` and `scripts/viz_cmap.py` are the diagnostics for this.

## Official method port (done, not in git)

- Full val run finished 2026-10-15, 50/50 scenes, about 11.5 min per scene; scores above.
- A worker agent ported the official SceneDiff pipeline (π³ + DINOv2 in place of the gated DINOv3) to MPS in `third_party/scenediff_mps`. Its config is `configs/scenediff_mps.yml` and its output goes to `out/pred_official`.
- π³ is the slow stage: about 340 s for 9 frames with the GPU shared.
- On this 16 GB Mac, run only one VGGT or π³ job at a time. Two at once fill swap and slow everything 5–10x.

## Official SceneDiff evaluation (`bench.py`)

- The script is `third_party/scene_diff/scripts/evaluate_multiview.py`. Clone https://github.com/yuqunw/scene_diff into `third_party/scene_diff`.
- `bench.official_eval` runs it with a cwd in which `data/scenediff_benchmark/splits` resolves (the script reads that path relative to cwd).
- `uv run scripts/gt_as_pred.py out/gt_pred` writes GT as predictions. Every metric comes out at 1.0, which confirms the scorer and the writer.
- **Format: follow the code, not the README.**
  - `{obj_id: {'video_1': {frame: {'mask': RLE, 'cost': float}}, 'video_2': {...}}, 'H', 'W'}`. Write it with `bench.save_pred`.
  - An object with masks in both videos counts as moved. An object with masks in one video counts as added or removed.
  - Frame index = video frame // 30. Masks are resized to a long side of 1024. Matching is by mask IoU > 0.5. Higher `cost` ranks first.
- Runs on CPU or MPS (no CUDA needed). Takes about 5 minutes for 50 scenes.

## Data and reproduction

- `uv sync`. ffmpeg must be on PATH. The VLM track needs the `codex` CLI, logged in.
- Data (not in git, 1.1 GB): `uv run scripts/fetch_scenediff.py --list`, then fetch pairs to `data/scenediff`.
- Gated, no access: `facebook/sam3`, DINOv3. Accessible: `yyfz233/Pi3`, `facebook/sam2.1-*`, DINOv2, VGGT.
- Old pipeline, one pair: `uv run diff.py <video1> <video2> -o out/val/<pair>`. Homemade score: `uv run eval.py out/val data/scenediff/scenediff_benchmark/data`.
- Tests: `uv run pytest -q` (12 pass).

## Not in git

- `data/` (SceneDiff videos and GT).
- `out/` (reconstructions, frames, predictions, VLM cache).
- `third_party/` (official repo and the MPS port).
- `examples/` (two personal test videos).
- `.venv/`.
