# Room change detection

Give it two RGB phone videos of the same room, taken at different times. It reports which objects were added, removed or moved, as a text report and an interactive 3D viewer.

## Setup

- `uv sync`
- ffmpeg on PATH (`brew install ffmpeg`)
- The first run downloads about 5 GB of model weights (VGGT-1B and CLIP ViT-L/14). After that the pipeline runs offline. The viewer loads three.js from a CDN, so it needs internet.
- Uses MPS when available, else CPU. Only MPS was measured (M4, 24 GB).

## Run

```
uv run diff.py A.mp4 B.mp4 -o out/
```

- `--frames N`: frames sampled per video, default 24. Both videos go through VGGT in one pass, so the model sees 2N images. The default is 48 images total, about 119 s on MPS. The spike ran it without running out of memory on a 24 GB M4. If memory runs out, lower `--frames`.
- Expected failures print one `error:` line and exit with code 1: missing file, ffmpeg not installed, portrait and landscape videos mixed, videos that do not overlap, no floor found.

## Recording the videos

- 20 to 30 s per video.
- Same room, same phone orientation for both videos (both landscape or both portrait).
- Slow pan. Floor and walls in view.
- Film B from a different spot.
- Between A and B, add, remove or move 2 to 3 objects.

If the result is wrong:

- Overlap error: re-record, with shared walls and floor in view.
- False additions at the room edges: change `diff(sa, sb)` in `diff.py` to `diff(sa, sb, tol=2)`.
- Missing small items: lower `min_voxels` in `blobs` in `changes.py`.
- Wrong labels: save the crop with `label.best_crop(...).save("crop.png")` and check it. The crop may be wrong, not CLIP.

## Output

- `out/report.md`: one line per change, with confidence.
- `out/changes.json`: the same changes with bounding boxes.
- `out/viewer.html`: 3D view. Switch between Diff, Video A and Video B. Green is added, red is removed, blue is moved. In Diff, the B cloud is grey where A never saw the space, and red ghost points show removed objects where they were in A.
- `out/frames/`: frames extracted from the videos.

## How it works

- Joint reconstruction: both videos go through VGGT in one pass, so both share one coordinate frame. There is no alignment step that can fail. The frame is rotated gravity-up and scaled to meters from an assumed camera height.
- Image-space diff (`change2d.py`): each frame's DINOv2 patches are compared with the same 3D spot in the other video's nearest-pose frames. Patches that look different vote into 5 cm voxels, and connected voxels become changed blobs. Space the other video never saw gets no vote.
- Free-space carving (`changes.py`) is still used for the overlap check and the viewer's grey "unseen" area.
- CLIP only names the changed blobs, zero-shot against a fixed vocabulary in `label.py`. A removed blob and an added blob with similar look and volume pair up as "moved".

## Limitations

- Changes under about 10 cm are not reliably found.
- The scale assumes a 1.4 m handheld camera height (`recon.CAM_HEIGHT`). Sizes are not reported.
- Left, right, front and back follow the reconstruction's own axes. They are consistent within one run but not compass directions.
- A moving person or pet shows up as a change. CLIP may label it "person".
- CLIP can mislabel unusual objects.
- Not in this version: sizes, custom vocabulary, crop gallery, GUI, server, object tracking.

## Tests

```
uv run pytest -q
```
