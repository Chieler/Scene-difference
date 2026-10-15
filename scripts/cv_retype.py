"""2-fold cross-validated choice of object_similarity_threshold for official SceneDiff (MPS port).

The 50 val/varied scenes are split in two halves (seeded shuffle). For each fold the threshold
maximizing obj/sc AP with type is chosen on one half and applied to the other. The held-out
halves are then scored together as one 50-scene prediction, so no scene is scored at a
threshold chosen on it. The faithful 0.7 result is the headline; this is a diagnostic.

  uv run python scripts/cv_retype.py out/pred_official out/retype_cv
"""
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402

GRID = [round(0.3 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90


def metrics(text):
    get = lambda pat: float(re.search(pat + r':.*?:\s*([0-9.]+)', text).group(1))
    return {'px_iou': get(r'Metric 1'), 'obj_im_ap': get(r'Metric 2'),
            'obj_sc_ap': get(r'Metric 3a'), 'obj_sc_ap_type': get(r'Metric 3b')}


def retype_and_score(src, work, scenes, t):
    d = Path(work) / f't{t:.2f}_{abs(hash(tuple(scenes))) % 10**6}'
    if d.exists():
        shutil.rmtree(d)
    subprocess.run([sys.executable, str(ROOT / 'scripts/retype_official.py'), str(src), str(d),
                    '--threshold', str(t), '--scenes', *scenes], check=True)
    return metrics(bench.official_eval(d, d / 'result.txt'))


def main():
    src, work = Path(sys.argv[1]), Path(sys.argv[2])
    scenes = sorted(p.name for p in src.iterdir() if (p / 'association.pkl').exists())
    rng = random.Random(0)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    halves = [sorted(shuffled[::2]), sorted(shuffled[1::2])]
    report = {'scenes': len(scenes), 'halves': halves, 'folds': []}
    final = work / 'cv_heldout'
    if final.exists():
        shutil.rmtree(final)
    for k, (fit, held) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        sweep = {t: retype_and_score(src, work, fit, t) for t in GRID}
        best = max(GRID, key=lambda t: (sweep[t]['obj_sc_ap_type'], sweep[t]['obj_sc_ap'], -abs(t - 0.7)))
        report['folds'].append({'fit_sweep': sweep, 'chosen': best,
                                'heldout_at_chosen': retype_and_score(src, work, held, best),
                                'heldout_at_0.7': retype_and_score(src, work, held, 0.7)})
        subprocess.run([sys.executable, str(ROOT / 'scripts/retype_official.py'), str(src), str(final),
                        '--threshold', str(best), '--scenes', *held], check=True)
        print(f'fold {k}: chosen {best}', report['folds'][-1]['heldout_at_chosen'], flush=True)
    report['cv_all_scenes'] = metrics(bench.official_eval(final, final / 'result.txt'))
    print('cross-validated (held-out halves together):', report['cv_all_scenes'])
    json.dump(report, open(work / 'cv_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
