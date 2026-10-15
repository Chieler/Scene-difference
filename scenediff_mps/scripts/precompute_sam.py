"""
MPS port: precompute stage-5 SAM masks for every scene in one process.

Writes the same <scene_dir>/sam_masks.pkl cache that SceneDiff.process_scene reads
(MaskModel.load_or_generate_masks), so predict_multiview.py then skips SAM. SAM is
~26 s/frame on an M2 and the model is loaded once here instead of once per scene.

Usage: python scripts/precompute_sam.py --config configs/scenediff_mps.yml --splits val --sets varied
"""

import os
import sys
import time
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image

from modules.mask_model import MaskModel
from predict_multiview import load_config, get_scenes_to_process, prepare_scene_data


def pipeline_hw(first_image, max_size=518):
    """(H, W) that utils.load_images_as_tensor_from_list produces."""
    W_orig, H_orig = Image.open(first_image).size
    if H_orig > W_orig:
        return max_size, round(round(W_orig * (max_size / H_orig)) / 14) * 14
    return round(round(H_orig * (max_size / W_orig)) / 14) * 14, max_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/scenediff_mps.yml')
    parser.add_argument('--splits', default='val')
    parser.add_argument('--sets', default='varied')
    parser.add_argument('--scenes', nargs='*', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    scenes = get_scenes_to_process(config, args)
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mask_model = MaskModel(config, device)

    for scene in scenes:
        scene_dir = Path(config['dataset']['gt_dir']) / scene
        cache = scene_dir / 'sam_masks.pkl'
        if cache.exists():
            print(f'{scene}: cached')
            continue
        img_1_list, img_2_list = prepare_scene_data(scene_dir, config['dataset']['resample_rate'])
        file_list = img_1_list + img_2_list
        H, W = pipeline_hw(file_list[0])
        t0 = time.time()
        mask_model.load_or_generate_masks(file_list, H, W, cache)
        print(f'{scene}: {len(file_list)} frames, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
