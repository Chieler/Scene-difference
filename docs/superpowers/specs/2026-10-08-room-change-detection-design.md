# Room Change Detection: MVP Design

## Goal

Show that the approach works: take two RGB phone videos of the same room, reconstruct them in one shared 3D frame, and report which objects were added, removed or moved, with labels. The MVP is a demonstration, not the full product.

Input is plain RGB video (no LiDAR). Everything runs locally on an Apple M4 (24 GB). No VLM or API calls.

## Pipeline

```
A.mp4, B.mp4 -> sample ~40 frames each (ffmpeg)
  -> VGGT joint run -> shared camera poses + depth + confidence
  -> floor-plane fit, rotate to gravity-up
  -> voxel grid (5 cm): per-video occupied / free / unknown
  -> diff -> blobs -> CLIP label -> added / removed / moved
  -> viewer.html + report.md
```

## Change logic (free-space carving)

- Each video marks voxels as occupied (depth hit), free (a camera ray passed through) or unknown (never seen).
- Added: occupied in B and known free in A. Removed: the reverse.
- A voxel unknown in the other video is never reported. This is how occlusion and viewpoint gaps avoid false changes. It is drawn grey in the viewer.
- Blobs are connected components of changed voxels, filtered by a minimum size (~10 cm).
- Moved: a removed blob paired with an added blob by similar volume and CLIP appearance similarity. Unpaired blobs stay added or removed.

## Labeling

For each blob, crop the best-view frame and run CLIP zero-shot against a built-in household vocabulary (~100 words). The top label goes into a template sentence, for example "Added: backpack near the desk". Location is described relative to the nearest labeled static object or the room extent.

## Alignment failure

If the joint reconstruction does not register the two videos (low pose confidence, or walls and floor do not coincide), the tool exits with "videos don't overlap enough, re-record with shared walls in view" and produces no diff.

## MVP scope

In:
- CLI: `uv run diff.py A.mp4 B.mp4 -o out/`
- Outputs: `viewer.html` (three.js, point cloud with colored boxes for added, removed and moved, plus A / B / diff toggle), `report.md`, `changes.json`
- `test_diff.py`: a synthetic scene (box room, fake poses) where a box is added, removed, moved and hidden from one camera. It asserts the added, removed, moved and unknown results. No model or GPU needed.
- `README.md`: setup, run instructions and a short note on tradeoffs
- `examples/`: the user's two videos and the resulting `out/`

Deferred (not in the MVP):
- `--sizes` and `--cam-height` metric scaling (sizes are optional; the MVP uses relative scale only)
- `--vocab` custom vocabulary and `--frames` flags
- Per-change crop gallery in the viewer
- MASt3R fallback
- Detection of changes under ~10 cm
- Real-time, GUI, server, training, tracking over more than two scans

## Files

- `diff.py`: CLI and pipeline
- `recon.py`: frames to shared cloud
- `changes.py`: voxels, free-space, blobs, moved pairing
- `label.py`: CLIP labels and sentences
- `viewer.py`: writes the HTML
- `test_diff.py`

## Order of work

1. Spike: confirm VGGT loads and runs on MPS (pin Python 3.12 via `uv` if 3.14 fails), and that it registers two videos in one frame.
2. `changes.py` and `test_diff.py` first, since they need no model.
3. `recon.py`, `label.py`, `viewer.py`.
4. Run on the real videos.

## Risks

- VGGT memory on MPS with 80 frames. Mitigation: batch, or lower frame count and resolution.
- Poor overlap between videos breaks registration. Mitigation: the failure check above.
- CLIP mislabels unusual objects. Accepted for the MVP.
