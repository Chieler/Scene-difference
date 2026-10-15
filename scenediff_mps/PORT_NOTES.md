# SceneDiff on Apple silicon (MPS port)

A copy of [yuqunw/scene_diff](https://github.com/yuqunw/scene_diff) (MIT, see `LICENSE` and
`NOTICE.md`), changed so it runs on a 16 GB M-series Mac without CUDA. Every edit in the code is
marked with a `MPS port:` comment. The two submodules are not vendored:
`scripts/setup_scenediff.sh` at the repo root clones them at pinned commits and applies
`patches/*.patch`.

## Run

```bash
cd scenediff_mps
PYTHONUNBUFFERED=1 uv run python scripts/predict_multiview.py --config configs/scenediff_mps.yml \
    --splits val --sets varied --output_dir ../out/pred_official [--scenes bed_3_bed_4 ...]
cd .. && uv run python bench.py out/pred_official
```

The run resumes on its own: scenes whose `object_masks.pkl` exists are skipped. This is the
original behaviour. Per-scene wall time is appended to `out/pred_official/timings.jsonl`.

Data (val runs only, not in git): `data/scenediff_benchmark/{data,splits}` here are symlinks. `data` points to
`out/scenediff_work/<scene>/`, which holds symlinks to the original videos and `segments.pkl`.
The pipeline's own files are written there and nowhere else: extracted frames, `sam_masks.pkl`
and the `.subsampled` marker.

## Results: SceneDiff val/varied, 50 scenes, official scorer (2026-10-15)

| | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP (type) |
|---|---|---|---|---|
| This port, faithful config (`object_similarity_threshold` 0.7) | 0.354 | 0.548 | 0.400 | 0.318 |
| 2-fold CV re-threshold (diagnostic, `scripts/cv_retype.py`) | 0.354 | 0.545 | 0.373 | 0.283 |

- Scene-level counts: 94 TP, 319 FP, 67 FN. All 50 scenes ran, 0 failures.
- Runtime on a 16 GB M2 with the machine to itself: 9.54 h for 1045 frames, about 33 s per frame, 89 to 1646 s per scene. SAM ViT-H takes about 26 s per frame of that. Per-scene times are in `out/pred_official/timings.jsonl`. bed_3 and laundry_3 had their SAM masks cached from debugging.
- CV: seeded split into two 25-scene halves. In each fold the threshold is chosen on one half (grid 0.30 to 0.90, maximising obj/sc AP with type) and applied to the other. The folds chose 0.4 and 0.7. Scored together, the held-out halves come out below the faithful 0.7, so 0.7 is not miscalibrated for DINOv2.
- The paper's 22.8 obj/sc AP is on SD-V **test**; these numbers are on **val**, so they are not directly comparable.

## Method deviations (these can change results)

| What | Original | Port | Why |
|---|---|---|---|
| Appearance backbone | DINOv3 ViT-H+/16 (`dinov3_vith16plus`, 1280-d) | DINOv2 ViT-L/14 (`facebook/dinov2-large` via `transformers`, 1024-d) | DINOv3 weights are gated (no access). Preprocessing is the same: ImageNet normalisation, resize to a multiple of the patch size, layer-normed patch tokens, bilinear upsampling to image size. The token grid on 518x392 inputs is 37x28 here instead of 33x25. |
| SAM input size | Full-resolution video frames (1920x1440 / 1920x1080) | Long side downscaled to 1024 (`models.sam.max_side`) with cv2 INTER_AREA. `min_mask_region_area` and the LangSplat `min_size=100` are scaled by the area ratio. | Full-resolution 64x3 mask logits plus the NMS took about 9 GB and 2 to 7 min per frame on this machine. SAM's encoder sees a 1024 px image either way, and its masks are nearest-resized to the 518 px Pi3 grid right afterwards. |

## Numerically neutral changes (same output, different device, speed or memory)

- `.cuda()` becomes `.to(self.device)` / `.to(DEVICE)`. Device order is CUDA, then MPS, then CPU, with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- `torch.cuda.get_device_capability` plus `autocast('cuda')` is replaced by `modules/device.py:autocast_dtype` and autocast on the current device type (bf16 on MPS, as on Ampere). Pi3's `autocast(device_type='cuda', enabled=False)` blocks use the tensor's device type.
- `torch_scatter.scatter_mean` is replaced by `modules/device.py:scatter_mean` (index_add sum / count; empty bins give 0, as torch_scatter does).
- `aggregate_features_to_voxels` accumulates per-image feature sums on the CPU. The original first stacked N x H x W x C features, about 30 GB.
- `_process_point_cloud_costs`: the per-voxel Python loop is now one vectorised index assignment.
- `SemanticModel.extract_features` caches patch tokens per frame. The pipeline re-extracts the same frame once per paired view; upsampling is still redone on every call.
- `utils.mask_nms`: the O(N²) Python loop over mask pairs is replaced by a chunked matmul for intersections. Counts stay below 2^24, so float32 is exact. Checked against the original on 120 random cases: identical keep sets.
- `merge_objects`: the faiss `IndexFlatL2` 1-NN search is replaced by an exact torch `cdist` (`utils.nearest_sq_dist`). On macOS, faiss-cpu and torch load two libomp copies, and the process aborts with OMP Error #15 at stage 8 (a silent exit).
- Region aggregation (`MaskModel.aggregate_to_regions`, `compute_rendering_cost`, and both loops in `get_dino_matched_region_cost`) is vectorised (`utils.region_mean_map`, `utils.region_feature_columns`) using one-hot matmuls. MPS `index_add_` serialises on the few large SAM regions (about 1 s per call). Checked against the original loops on 200 random cases: max abs diff 3.6e-7.
- DINO features: the token cache plus `utils.feature_chunks` upsamples in 128-channel slices, and `reprojected_cosine_distance` fuses grid_sample and cosine per slice. The 0.8 GB full-resolution 1024-channel map is never built. Max abs diff 1.2e-7 against the original path.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled only on CUDA. MPS has no deterministic kernels for most ops, and the substitutes made stages 6 to 8 3 to 10x slower. bed_3_bed_4 output (masks and costs) is identical with and without the stage-6 changes.
- `scripts/precompute_sam.py` (optional) fills the same `sam_masks.pkl` cache for all scenes, with SAM loaded once.
- SAM (submodule copy): float32 point tensors (MPS has no float64); `MaskData.filter` uses index_select; RLE encoding runs on CPU; `points_per_batch` 64 → 16 (memory only).
- `mask_model._initialize_sam`: re-import after the `sys.path` fallback. The original's fallback path never imported.
- `open3d` is imported lazily (only `vis_pc` needs it). `save_detections` (per-frame PNGs) is off.
- `predict_multiview.py`: adds `--scenes`, per-scene try/except and timings. After extraction it keeps only every 30th frame on disk (the same frames `[::30]` would pick).
