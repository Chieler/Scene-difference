# Handoff (2026-10-14)

## Goal

Given a before video and an after video of a room, find objects that were added, removed or moved. For each one, output a per-frame object mask and a 3D position. The benchmark is [SceneDiff](https://github.com/yuqunw/scene_diff) ([paper](https://arxiv.org/abs/2512.16908)).

## Current pipeline (`diff.py`)

1. **VGGT-1B** (`recon.py`) reconstructs both videos in one forward pass (24 frames each), so they share one coordinate frame. The frame is rotated gravity-up with a RANSAC floor fit and scaled to meters from an assumed 1.4 m camera height.
2. **Free-space carving** (`changes.py`, 5 cm voxels) checks that the videos overlap and feeds the viewer. It no longer detects changes.
3. **2D patch diff** (`change2d.py`) does the detection:
   - DINOv2-base patch features at 2x upscale.
   - Each patch is reprojected through VGGT depth into the 8 nearest-pose frames of the other video, with an occlusion check.
   - Change score = 1 - best similarity in a 3x3 window.
   - Changed patches vote into voxels, and connected voxels become blobs (`VOTE` in `diff.py`).
4. **CLIP ViT-L/14** (`label.py`) names the blobs. `pair_moved` pairs a removed blob and an added blob into "moved".

Output: 3D boxes in `changes.json`, a report, and a three.js viewer. There are no masks yet.

## Results on SceneDiff val (50 pairs, homemade scorer)

`eval.py` projects each predicted 3D box into the frames and matches it to the GT mask's 2D box at IoU >= 0.2. This is looser than the official metric.

| Version | Precision | Recall | F1 |
|---|---|---|---|
| 3D voxel diff (old) | | | ~0.16 |
| 2D patch diff (current) | 0.29 | 0.26 | 0.27 |

Type-correct: 25 of 161 objects.

Diagnostics (`scripts/check2d.py`):
- Raw patch AUROC (changed vs unchanged) is 0.82.
- 5.5% of unchanged patches fire.
- False-alarm area is about 5x the real change area.

## Why this approach hits a ceiling

- VGGT depth disagrees between the two videos by 10 to 15 cm locally.
- The median missed object covers 0.9% of the frame.
- Geometry noise is therefore as large as the objects. Any voxel diff or reprojected patch diff floods with false alarms.
- More tuning will not fix this.

## Recommended next step: object-level pipeline

Segment objects first, then compare objects, not pixels or voxels:

1. **Propose objects.** Two options:
   - A VLM looks at about 16 frames per video and lists candidate changes.
   - Ungated alternative: SAM2 automatic masks, or Grounding DINO / OWLv2 with a generic vocabulary.
2. **Segment and track** each object through both videos. SAM 3 (`facebook/sam3`, gated) segments by text prompt. The ungated alternative is SAM2 video. All of these are in the installed `transformers`.
3. **Place each instance in 3D** with VGGT depth (mask centroid lifted to world coordinates), and embed it with DINO.
4. **Match instances across videos** by 3D position and embedding. An unmatched instance in A is removed, an unmatched instance in B is added, and the same instance at a different position is moved.
5. **Score with the official SceneDiff evaluation** (next section), which gives real AP.

Open decisions for the user:
- Which VLM and API key to use.
- Whether to accept the SAM 3 license on Hugging Face and run `hf auth login`.
- Fully local alternative: Qwen2.5-VL + Grounding DINO + SAM2.

3D object reconstruction (SAM 3D Objects, object-level Gaussian splatting) is not needed for masks and positions. It is heavy and CUDA-oriented. It may be useful later, for example to get grasp shapes for objects that were already found.

## Official SceneDiff evaluation

The script is `third_party/scene_diff/scripts/evaluate_multiview.py`. To get it, clone https://github.com/yuqunw/scene_diff into `third_party/scene_diff`. It has not been run here yet.

- Predictions go in `<pred_dir>/<scene>/object_masks.pkl`.
- **Format: follow the code, not the README.** `extract_detections` expects `{obj_id: {'video_1': {frame: {'mask': RLE, 'cost': float}}, 'video_2': {...}}}`, plus top-level `'H'` and `'W'` keys, which it skips.
- An object with entries in both videos counts as moved. An object with entries in only one counts as added or removed.
- GT frames are every 30th video frame (`resample_rate=30`): video frame f maps to GT index f // 30.
- `max_length=1024`. Matching is by mask IoU > 0.5.
- Metrics: pixel IoU, per-frame AP and per-scene AP, both type-agnostic and type-aware.
- Needs torch, cv2, matplotlib, pycocotools. Check that it runs without CUDA.

Do not score the current box blobs with it. Boxes against a mask-IoU threshold of 0.5 score near zero and tell nothing new.

## Data and reproduction

- `uv sync`. ffmpeg must be on PATH.
- Data (not in git, 1.1 GB): `uv run scripts/fetch_scenediff.py --list`, then fetch pairs to `data/scenediff`.
- Run one pair: `uv run diff.py <video1> <video2> -o out/val/<pair>`. VGGT output is cached in `recon.npz`.
- Homemade score: `uv run eval.py out/val data/scenediff/scenediff_benchmark/data`.
- Vote threshold sweep: `uv run scripts/sweep2d.py`. It caches scores in `out/s2d`.
- Raw 2D signal quality: `uv run scripts/check2d.py`.
- Tests: `uv run pytest -q` (12 pass).

## Not in git

- `data/` (SceneDiff videos and GT).
- `out/` (4.2 GB: reconstructions, results).
- `third_party/`.
- `examples/` (two personal test videos).
- `.venv/`.
