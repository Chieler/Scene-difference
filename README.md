# What changed in the room?

Give it two phone videos of the same room, recorded at different times. It builds one shared 3D model from both, and finds objects that were added, removed or moved. Each change gets a name, a sentence, a 3D box, before and after pictures, and a verdict from geometry on whether the change is real.

```
uv run diff.py before.mp4 after.mp4 -o out/
```

<!-- EXAMPLE: replace with the real run on examples/ -->
> **Example** (`examples/before.mp4` → `examples/after.mp4`): _to be filled in with the committed run in `examples/out/`._

## The idea: a VLM proposes, geometry disposes

A vision-language model is very good at noticing *that* something changed and *what* it is ("a green bottle with a black cap"). It is bad at telling a real change from one caused by a new viewpoint: an object hidden behind a chair in one video looks "removed" to it.

3D geometry has the opposite profile. Once both videos share one coordinate frame, you can ask exactly *where* the other video looked and *what it saw there*. But pixel-level geometry diffs drown in noise. Monocular depth from two separate videos disagrees by 10 to 15 cm, as much as the objects themselves. My first, geometry-only version of this project hit that ceiling (F1 0.27 on the SceneDiff benchmark).

So the two split the work:

1. **Shared 3D.** [VGGT](https://github.com/facebookresearch/vggt) reconstructs both videos in a *single* forward pass (24 frames each), so both come out in one coordinate frame. No separate registration step can fail. The frame is rotated gravity-up and scaled to meters from a RANSAC floor fit.
2. **Propose.** An OpenAI vision model sees both videos as labeled contact sheets. For each change it returns a name, a sentence, a type and a box in every frame where the object appears.
3. **Segment and lift.** SAM2 turns every box into a mask. VGGT depth lifts each mask into the shared 3D frame.
4. **Verify against the other video.** For each claimed object, project its 3D points into every frame of the *other* video, and check visibility with a depth test:
   - **not seen:** the other video never looked at that spot, or something nearer was in the way. Then the claim is *unverified*, not *removed*. Incomplete observation is not evidence of change.
   - **seen through:** the other video sees *past* the spot, into free space where the object stood. That is the strongest evidence of a change.
   - **looks different:** DINOv2 features at the spot don't match the object.

   The object is scored *relative to a ring of background around it*. The ring has the same registration noise, so only evidence above the noise floor counts. Claims the geometry contradicts are *rejected* but kept in the output, so you can see what the VLM got wrong. A "moved" object whose two 3D positions coincide is also rejected.
5. **Present.** The output has a report, a before/after crop per change, and an interactive 3D viewer. The viewer shows boxes, the object's own points, a ghost of removed objects, a toggle for rejected claims, and grey where the other video never looked.

No API key? `--detector geometry` runs the older geometry-only detector (DINO patch diff voted into voxels, labeled by CLIP). It is weaker, but fully local.

## Setup

- `uv sync`
- ffmpeg on PATH (`brew install ffmpeg`)
- `export OPENAI_API_KEY=...`. Optional: `OPENAI_MODEL` (default `gpt-5`).
- The first run downloads about 7 GB of weights: VGGT-1B, SAM2.1-base+, DINOv2-base, CLIP ViT-L/14. Uses MPS when available, else CPU. Tested on an M-series Mac with 16 GB.

## Output

- `out/report.md`: confirmed, unverified and rejected changes, each with its reason and a before/after picture.
- `out/viewer.html`: the 3D view. Open it in a browser (it loads three.js from a CDN). To see the before/after pictures in its side panel, serve the folder: `python -m http.server -d out`. Click a change to fly to it.
  - Green = added, red = removed, blue = moved, yellow = unverified.
  - Grey points = space the other video never saw.
- `out/changes.json`: everything above, machine-readable: 3D boxes, evidence numbers, frames.
- `out/changes/NN.jpg`: before/after crops with the object outlined.
- `out/vlm.json`: the raw VLM claims, cached, so reruns are free.

## Recording tips

- 20 to 30 s per video, slow pan, same phone orientation for both.
- Keep the floor in view for metric scale. Without it, positions are in VGGT units.
- Film the second video from a different spot. Between videos, add, remove or move 2 to 3 objects.

## How well does it work? (SceneDiff benchmark)

[SceneDiff](https://github.com/yuqunw/scene_diff) has 50 validation pairs of real room videos with per-frame masks of every changed object. I scored with its official evaluator (`bench.py` wraps `third_party/scene_diff/scripts/evaluate_multiview.py`):

| Method | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP + type |
|---|---|---|---|---|
| Geometry only: VGGT + DINOv2 object diff (`objdiff.py`), 1 scene | 0.008 | 0.024 | 0.125 | |
| VLM proposals + SAM2 masks (`scripts/vlm_track.py`) | 0.520 | 0.322 | 0.323 | 0.284 |
| Official SceneDiff method, ported to MPS with DINOv2 in place of gated DINOv3 | 0.354 | 0.548 | 0.400 | 0.318 |
| Fusion of the two (`scripts/fuse.py`) | 0.490 | 0.576 | 0.498 | 0.448 |
| **Fusion + VLM verifier, rejected candidates ranked last** (`scripts/verify.py`) | 0.490 | **0.585** | **0.522** | **0.472** |
| Fusion + VLM verifier, rejected candidates dropped | **0.605** | 0.548 | 0.494 | 0.449 |

All rows: val split, "varied" set, 50 scenes, except where noted. The fusion and verifier settings were chosen on these same scenes, so treat the last three rows as optimistic until they are confirmed on the test split.

The two methods fail in opposite ways. The official method segments everything and keeps what geometry calls changed: high recall, but 319 false alarms. The VLM names few objects, and is usually right, but misses small ones. The fusion ranks objects both found first, then the official method's other objects, then the VLM's. The verifier then shows the VLM each object only the official method found, as an outlined crop next to both videos, and asks if it is a real change. It keeps 64 of 331; dropping the rest cuts false alarms from 348 to 91.

The VLM row is the proposal stage of this pipeline, run through the `codex` CLI on the benchmark's own frames. The paper's own numbers are on its test split, so they are not directly comparable. To reproduce:
```
uv run scripts/fetch_scenediff.py --list      # then fetch the val pairs to data/scenediff
uv run scripts/vlm_track.py                   # -> out/pred_vlm/<scene>/object_masks.pkl
uv run scripts/locate.py out/pred_vlm         # 3D position per object -> objects.json
uv run python bench.py out/pred_vlm           # official scorer
# official method: see third_party/scenediff_mps/PORT_NOTES.md (about 7.5 h for 50 scenes on a 16 GB Mac)
uv run scripts/fuse.py                        # fusion variants, scored
uv run scripts/verify.py                      # VLM verifier on the official-only objects, scored
```

## Design choices and tradeoffs

- **One VGGT pass for both videos, instead of two reconstructions plus registration.** Registration between separately reconstructed videos is the classic failure point, and joint inference removes it. The cost is memory, since all frames go through the model at once; 48 frames fit in 16 GB.
- **Proposals from a VLM, not from pixel diffs.** Pixel and voxel diffs are dominated by depth noise, and they also lack names. The VLM gives semantics for free, and its claims are concrete enough (a box per frame) to check.
- **Verification uses the other video's actual rays, not a voxel grid.** "Not seen" vs "seen through" vs "looks the same" are three different answers. Only rays the other video actually cast can tell them apart. Comparing the object against its own background ring cancels most of the registration error.
- **Rejected claims stay in the output.** You can see what the VLM claimed and why geometry disagreed, which makes the system's behavior inspectable.
- **Limits.**
  - Objects under about 0.5% of the frame are too small to test in 3D, and are reported as unverified.
  - A person walking through can be reported as a change.
  - The API call costs a few cents per run, and the VLM can miss small objects. On the benchmark, misses outnumber false alarms 2:1.

## Code

| File | Role |
|---|---|
| `diff.py` | CLI: reconstruction cache, detector choice, report, viewer |
| `recon.py` | VGGT joint reconstruction, floor alignment |
| `semantic.py` | VLM proposals, SAM2 masks, 3D lifting, geometric verification, crops |
| `change2d.py`, `changes.py`, `label.py` | Geometry-only detector: DINO patch diff, free-space carving, CLIP labels |
| `viewer.py` | Self-contained three.js viewer |
| `bench.py`, `scripts/` | SceneDiff benchmark: frame extraction, official scorer, VLM track, 3D placement |
| `objdiff.py` | Geometry-first object pipeline, kept as the negative result in the table |

Tests: `uv run pytest -q`
