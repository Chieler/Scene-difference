"""Re-run the cross-video association of official SceneDiff (MPS port) at another threshold.

Uses the association.pkl that third_party/scenediff_mps writes next to object_masks.pkl: the
object similarity matrix and the per-video masks from before the merge. This replays
SceneDiff._create_unified_object_ids and _save_object_masks without running any model.

  uv run python scripts/retype_official.py out/pred_official out/pred_retype_0.5 --threshold 0.5
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def unify(assoc, threshold):
    """SceneDiff._create_unified_object_ids: video_2 objects reuse a video_1 id if max sim > t."""
    sim = assoc['object_sim_matrix']
    ids_1 = [int(i) for i in assoc['object_ids_1']]
    unit = {'video_1': {i: n for n, i in enumerate(ids_1)}, 'video_2': {}}
    next_id = len(ids_1)
    for obj_id in (int(i) for i in assoc['object_ids_2']):
        scores = sim[:, obj_id] if sim.numel() else None
        if scores is not None and scores.max() > threshold:
            unit['video_2'][obj_id] = unit['video_1'][ids_1[int(scores.argmax())]]
        else:
            unit['video_2'][obj_id] = next_id
            next_id += 1
    return unit


def merge_frame(old, new):
    """Union of two per-frame entries, as _save_object_masks does when ids collide. Masks of
    different objects in one frame are disjoint (one mask index per pixel), so the
    area-weighted cost equals the mean cost over the union."""
    a, b = mask_utils.decode(old['mask']).astype(bool), mask_utils.decode(new['mask']).astype(bool)
    union = a | b
    rle = mask_utils.encode(np.asfortranarray(union.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    na, nb = old['mask_size'], new['mask_size']
    return {'mask': rle, 'mask_size': int(union.sum()),
            'cost': (old['cost'] * na + new['cost'] * nb) / max(na + nb, 1)}


def retype(pred, assoc, threshold):
    unit = unify(assoc, threshold)
    out = {'H': pred['H'], 'W': pred['W']}
    for vk in ('video_1', 'video_2'):
        for obj_id, entry in assoc['masks_by_video'][vk].items():
            uid = unit[vk][obj_id]
            frames = out.setdefault(uid, {}).setdefault(vk, {})
            for f, d in entry[vk].items():
                frames[f] = merge_frame(frames[f], d) if f in frames else dict(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--threshold', type=float, required=True)
    ap.add_argument('--scenes', nargs='*', default=None)
    a = ap.parse_args()
    for d in sorted(Path(a.src).iterdir()):
        if not (d / 'association.pkl').exists() or (a.scenes and d.name not in a.scenes):
            continue
        pred = pickle.load(open(d / 'object_masks.pkl', 'rb'))
        assoc = pickle.load(open(d / 'association.pkl', 'rb'))
        (Path(a.dst) / d.name).mkdir(parents=True, exist_ok=True)
        with open(Path(a.dst) / d.name / 'object_masks.pkl', 'wb') as f:
            pickle.dump(retype(pred, assoc, a.threshold), f)


if __name__ == '__main__':
    main()
