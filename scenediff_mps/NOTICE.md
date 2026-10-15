# Third-party code and weights

| Component | Where it comes from | License | In this repo |
|---|---|---|---|
| SceneDiff method (`modules/`, `scripts/`, `utils.py`, `configs/`, `splits/`) | [yuqunw/scene_diff](https://github.com/yuqunw/scene_diff) @ `62dcbd2`, Yuqun Wu et al., *SceneDiff: A Benchmark and Method for Multiview Object Change Detection* (ECCV 2026) | MIT, `LICENSE` | Vendored with the MPS changes listed in `PORT_NOTES.md` |
| Pi3 (`submodules/Pi3`) | [yyfz/Pi3](https://github.com/yyfz/Pi3) @ `0a505af` | Pi3 license, **non-commercial** | Not vendored. `scripts/setup_scenediff.sh` clones it and applies `patches/Pi3.patch` |
| SAM, LangSplat variant (`submodules/segment-anything-langsplat-modified`) | [yuqunw/segment-anything-langsplat-modified](https://github.com/yuqunw/segment-anything-langsplat-modified) @ `100f1ca`, derived from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Meta's files are Apache-2.0. The fork has no license file | Not vendored. Cloned and patched by the setup script (`patches/segment-anything-langsplat-modified.patch`) |
| SAM ViT-H checkpoint `sam_vit_h_4b8939.pth` | `dl.fbaipublicfiles.com/segment_anything/` | Apache-2.0 | Downloaded by the setup script into `checkpoints/` |
| Pi3 weights `yyfz233/Pi3` (tested at `ae722e7`) | Hugging Face | See the model card | Downloaded by the setup script into the HF cache |
| DINOv2 ViT-L/14 `facebook/dinov2-large` (tested at `47b73ee`) | Hugging Face | Apache-2.0 | Downloaded by the setup script into the HF cache |

The upstream `submodules/dinov3` is not used. DINOv2-L replaces it (see `PORT_NOTES.md`).
Because Pi3 is non-commercial, the SceneDiff candidate path is for research and evaluation use only.
