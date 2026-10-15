#!/bin/bash
# Official SceneDiff (MPS port) on SceneDiff val/varied, then the official scorer.
# Resumable: scenes with out/pred_official/<scene>/object_masks.pkl are skipped.
# Usage: scenediff_mps/run_val.sh [scene ...]
set -e
cd "$(dirname "$0")"
mkdir -p ../out/logs
PYTHONUNBUFFERED=1 uv run python scripts/predict_multiview.py --config configs/scenediff_mps.yml \
    --splits val --sets varied --output_dir ../out/pred_official ${1:+--scenes "$@"} \
    2>&1 | grep --line-buffered -v -E "Warning|warnings.warn|kthvalue" >> ../out/logs/official_val.log
cd ..
uv run python bench.py out/pred_official
