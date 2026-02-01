#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_LIST=(
  "$ROOT_DIR/trainer_depends/configs/stage1_visual_encoder_wingtra.yaml"
    "$ROOT_DIR/trainer_depends/configs/stage1_visual_encoder_visloc.yaml"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="/home/data/zwk/pyproj_pylib_zwk/raster_handler:${PYTHONPATH:-}"

for CFG in "${CFG_LIST[@]}"; do
  python "$ROOT_DIR/trainers/stage1_visual_encoder_w_ANCE.py" --p_yaml "$CFG"
done
