# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Judges and evaluators** reviewing computer vision, spatial AI, and robotics submissions (e.g. Humanoids Challenge).
- **ML researchers and 3D vision practitioners** evaluating scene difference, spatial memory, change detection, and video-based scene understanding.
- **Robotics engineers** who require fast, offline-capable verification of real-world environment changes from handheld or onboard video trajectories.

## Product Purpose

Identify what changed in a room between two handheld mobile phone videos recorded at different times. The system detects additions, removals, and 3D object displacements, verifies candidate claims geometrically in a shared metric 3D model (carving free space and measuring DINOv2 appearance residuals), and renders natural-language explanations alongside an interactive 3D WebGL viewer.

## Positioning

Unlike pure 2D vision-language approaches (which hallucinate changes from perspective shifts or lighting) and pure geometric diffing (which drowns in point cloud noise and false positives), this system uses a hybrid fusion architecture:
1. Joint dual-video reconstruction (VGGT) in a single feed-forward pass aligning both videos into a shared gravity-aligned metric coordinate system without iterative registration.
2. Cross-video 3D ray verification (carving free space and checking feature residuals) that demotes or filters candidate false alarms.
3. State-of-the-art benchmark performance on the official SceneDiff validation split: **0.585 obj/im AP** (+3.7 over official SceneDiff) and **0.605 px/im IoU** (+25.1 points, cutting false alarms by 71.5%).
4. Dual runtime paths: a high-precision online VLM mode and a 100% local, offline CLIP+MPS mode requiring zero API keys.

## Operating Context

- Handheld smartphone video recordings of indoor rooms (bedrooms, bathrooms, living spaces).
- Runs on local Apple Silicon hardware (MPS-optimized) and Linux GPU environments.
- Outputs include structured `changes.json`, summary `report.md`, outlined before/after crop galleries, and standalone zero-dependency Three.js 3D viewers.

## Capabilities and Constraints

- **Input:** Two casual MP4 videos (typically 5–15 seconds, 1080p).
- **Metric measurements:** Ground-plane RANSAC floor detection calculates real-world scale and physical object displacement in meters (e.g., moved 3.10 m).
- **Classification categories:** `added`, `removed`, `moved`.
- **Status tiers:** `confirmed` (multi-view / 3D verified), `unverified` (occluded or too small to resolve geometrically), `rejected` (geometric rays or VLM verifier prove consistency).
- **Dependencies:** Python $\ge$ 3.10 with `uv`, `ffmpeg`, optional OpenAI API key for online VLM mode.

## Evidence on Hand

- Evaluated across all 50 scenes of the official SceneDiff validation split (`varied` subset) scored with the official competition evaluator (`bench.py`).
- 2-fold cross-validation (`scripts/cv_select.py`) independently selecting optimal thresholding without overfitting.
- Live interactive WebGL 3D demo hosted on GitHub Pages: [`https://chieler.github.io/Scene-difference/`](https://chieler.github.io/Scene-difference/) and offline variant.
- Headline example with before/after crops, metric vectors, and visual verification assets in `docs/` and `examples/`.

## Product Principles

1. **Geometry grounds semantics:** Never trust 2D pixel diffs or VLM interpretations without physical multi-view ray verification in 3D.
2. **Honest uncertainty:** Clearly categorize changes into Confirmed, Unverified (too small or unobserved), and Rejected (proven false alarms) rather than presenting a black-box binary score.
3. **Zero-friction inspection:** Provide immediate browser-based 3D interaction without requiring evaluators to install local dependencies to see results.
4. **Offline first-class:** Support a completely local, self-contained pipeline alongside the cloud VLM track.
