# What changed in the room?

Give it two phone videos of the same room, recorded at different times. It builds one shared 3D model from both, and finds objects that were added, removed or moved. Each change gets a name, a sentence, a 3D box, before and after pictures, and a verdict from geometry on whether the change is real.

Two versions, same output:

```
uv run diff.py before.mp4 after.mp4 -o out/            # full: uses a VLM (OpenAI API key, or the codex CLI)
uv run diff.py before.mp4 after.mp4 -o out/ --offline  # backup: fully local, no API, no network after setup
```

Without `--offline` it picks the full version when a VLM is available and falls back to offline otherwise.

<!-- EXAMPLE: replace with the real run on examples/ -->
> **Example** (`examples/before.mp4` → `examples/after.mp4`): _to be filled in with the committed run in `examples/out/`._

## The idea: two detectors that fail in opposite ways, and geometry as the judge

**Geometry** finds changes everywhere, including ones that aren't real. Once both videos share one coordinate frame you can ask exactly *where* the other video looked and *what it saw there*. But monocular depth from two separate videos disagrees by 10 to 15 cm, as much as the objects themselves, so pure geometry diffs drown in noise. The official [SceneDiff](https://github.com/yuqunw/scene_diff) method copes better than anything I built: it segments every frame with SAM and scores each segment by depth and appearance changes across many view pairs. On its benchmark it finds most changed objects, along with about three false alarms for each real change.

**A vision-language model** is the mirror image. It is very good at noticing *that* something changed and *what* it is ("a green bottle with a black cap"), and it rarely invents changes. But it misses small objects, and it can't tell a real change from one caused by a new viewpoint.

So the pipeline lets each cover the other:

1. **Shared 3D.** [VGGT](https://github.com/facebookresearch/vggt) reconstructs both videos in a *single* forward pass (24 frames each), so both come out in one coordinate frame. No separate registration step can fail. The frame is rotated gravity-up and scaled to meters from a RANSAC floor fit.
2. **Candidates from geometry.** The SceneDiff method (ported to Apple-silicon GPUs, `official.py`) runs on the same 24 + 24 frames and returns per-frame masks of objects that may have changed.
3. **Claims from the VLM** (full version). The VLM sees both videos as labeled contact sheets and returns, for each change, a name, a sentence, a type and a box in every frame where it appears. SAM2 turns the boxes into masks.
4. **Fuse.** A candidate whose masks overlap a VLM claim is the same object: it is ranked first and named by the VLM. Each remaining candidate is shown to the VLM as an outlined crop, next to both videos, and named or vetoed. In the offline version, CLIP names the candidates instead.
5. **Check every object in 3D against the other video.** VGGT depth lifts each mask into the shared frame. Its points are projected into every frame of the *other* video and tested for visibility:
   - **not seen:** the other video never looked at that spot, or something nearer was in the way. Then the claim is *unverified*, not *removed*. Incomplete observation is not evidence of change.
   - **seen through:** the other video sees *past* the spot, into free space where the object stood. That is the strongest evidence of a change.
   - **looks different:** DINOv2 features at the spot don't match the object.

   The object is scored *relative to a ring of background around it*. The ring has the same registration noise, so only evidence above the noise floor counts. Rejected objects are kept in the output, with the reason, so you can see what was ruled out and why. A "moved" object whose two 3D positions coincide is also rejected.
6. **Present.** The output has a report, a before/after crop per change, and an interactive 3D viewer. The viewer shows boxes, the object's own points, a ghost of removed objects, a toggle for rejected claims, and grey where the other video never looked.

On SceneDiff, fusing the two detectors scores higher than either alone: obj/im AP 0.585, against 0.548 for the official method and 0.322 for the VLM (table below).

Other modes: `--detector fast` runs only the VLM claims plus the 3D check (about 2 minutes, no SceneDiff stage). `--detector geometry` runs my first, voxel-diff detector.

## Setup

- `uv sync`
- ffmpeg on PATH (`brew install ffmpeg`)
- Full version, one of:
  - `export OPENAI_API_KEY=...`, optional `OPENAI_MODEL` (default `gpt-5`);
  - or the [codex CLI](https://github.com/openai/codex), logged in. It is used when no key is set.
- `bash scripts/setup_scenediff.sh` (both versions need it). It clones π³ and the SceneDiff SAM fork at pinned commits and applies the MPS patches from `scenediff_mps/patches/`. It also downloads the SAM ViT-H checkpoint (2.6 GB, sha256-checked) and the π³ and DINOv2-L weights. After that the SceneDiff stage runs without network.
- The first run downloads the model weights: VGGT-1B, π³, SAM ViT-H, SAM2.1-base+, DINOv2-base and -large, CLIP ViT-L/14. Uses MPS when available, else CPU. Tested on an M-series Mac with 16 GB.
- Run time on a 16 GB M2 for a pair of 30 s videos (SceneDiff `bathroom_1_bathroom_2`): reconstruction about 5.5 min, SceneDiff stage about 22 min, then the 3D check (offline, about 2 min) or VLM calls plus the check (full, about 3.5 min). Total is about 30 minutes either way; reruns use the caches and take about a minute. Peak memory is 12.7 GB, because the SceneDiff stage runs in its own process and frees its memory before the rest continues. `--detector fast` takes about 8 minutes.

## Output

- `out/report.md`: confirmed, unverified and rejected changes, each with its reason and a before/after picture.
- `out/viewer.html`: the 3D view. Open it in a browser (it loads three.js from a CDN). To see the before/after pictures in its side panel, serve the folder: `python -m http.server -d out`. Click a change to fly to it.
  - Green = added, red = removed, blue = moved, yellow = unverified.
  - Grey points = space the other video never saw.
- `out/changes.json`: everything above, machine-readable: 3D boxes, evidence numbers, frames.
- `out/changes/NN.jpg`: before/after crops with the object outlined.
- `out/vlm.json`, `out/verify.json`, `out/official/`: the raw VLM claims, the VLM's verdicts on the SceneDiff candidates, and the SceneDiff candidates, all cached, so reruns are fast.

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

All rows: val split, "varied" set, 50 scenes, except where noted. The variant was chosen by 2-fold cross-validation (`scripts/cv_select.py`): each half picks "ranked last", so the held-out score is the same 0.585. The fusion match threshold (mask IoU 0.3) was set once and not tuned.

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
- **Two detectors instead of one better one.** Geometry-first detection (SceneDiff) has high recall and many false alarms. The VLM has high precision, names everything, and misses small objects. Fusing them beat each alone on the benchmark. The VLM also checks the geometry's leftovers: it is much more reliable when asked "is this outlined thing a real change?" than "find every change in this room".
- **An offline version as a fallback, not a toy.** It uses the same candidates and the same 3D check; only the names come from CLIP instead of the VLM. The changes it finds are the SceneDiff method's (obj/im AP 0.548 on the benchmark). What it loses is the VLM's veto on false alarms and its sentences.
- **Verification uses the other video's actual rays, not a voxel grid.** "Not seen" vs "seen through" vs "looks the same" are three different answers. Only rays the other video actually cast can tell them apart. Comparing the object against its own background ring cancels most of the registration error.
- **Rejected claims stay in the output.** You can see what the VLM claimed and why geometry disagreed, which makes the system's behavior inspectable.
- **Limits.**
  - Objects under about 0.5% of the frame are too small to test in 3D, and are reported as unverified.
  - A person walking through can be reported as a change.
  - The SceneDiff stage takes about 20 minutes on a 16 GB Mac, mostly SAM ViT-H on 48 frames. `--detector fast` skips it (about 2 minutes) at lower recall.
  - In the offline version, "moved" comes from SceneDiff's own appearance match between videos. It can pair the object with a different, similar-looking one nearby, so the before picture may show the wrong object. The full version names and checks these with the VLM.
  - It uses π³, which is licensed for non-commercial use only (`scenediff_mps/NOTICE.md`).
  - The VLM calls cost a few cents per run with the OpenAI API.

## Code

| File | Role |
|---|---|
| `diff.py` | CLI: reconstruction cache, detector choice, report, viewer |
| `recon.py` | VGGT joint reconstruction, floor alignment |
| `semantic.py` | VLM claims, SAM2 masks, 3D lifting, the 3D check against the other video, crops |
| `official.py` | The SceneDiff method on the sampled frames: candidate masks |
| `fusion.py` | Full and offline versions: fusing candidates with VLM claims, VLM verification or CLIP names |
| `vlm.py` | VLM backend (OpenAI API or codex CLI), crop-sheet verifier |
| `change2d.py`, `changes.py`, `label.py` | Geometry-only detector: DINO patch diff, free-space carving, CLIP labels |
| `viewer.py` | Self-contained three.js viewer |
| `bench.py`, `scripts/` | SceneDiff benchmark: frame extraction, official scorer, VLM track, 3D placement |
| `objdiff.py` | Geometry-first object pipeline, kept as the negative result in the table |
| `scenediff_mps/`, `scripts/setup_scenediff.sh` | The SceneDiff method (MIT), ported to Apple-silicon GPUs; setup clones and patches its dependencies |

Tests: `uv run pytest -q`
