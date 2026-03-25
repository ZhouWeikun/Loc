#!/usr/bin/env bash
set -euo pipefail

# Sequential launcher for Stage-1 / Stage-2 override runners.
#
# Usage:
#   bash scripts/run_multi_stages.sh
#   bash scripts/run_multi_stages.sh stage1
#   bash scripts/run_multi_stages.sh stage2
#   bash scripts/run_multi_stages.sh both
#
# Optional env vars:
#   PYTHON_BIN=/root/miniconda3/envs/neuloc_wisp/bin/python
#   STAGE1_CFG=gen_fm_exps/run_yamls/stage1_cfg_var.txt
#   STAGE2_CFG=gen_fm_exps/run_yamls/stage2_cfg_var.txt
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE1_CFG="${STAGE1_CFG:-gen_fm_exps/run_yamls/stage1_cfg_var.txt}"
STAGE2_CFG="${STAGE2_CFG:-gen_fm_exps/run_yamls/stage2_cfg_var.txt}"
MODE="${1:-both}"
DRY_RUN="${DRY_RUN:-0}"

run_stage() {
  local stage="$1"
  local cfg_var_file="$2"
  local runner=""

  case "${stage}" in
    stage1)
      runner="${PROJECT_ROOT}/scripts/run_stage1_with_overrides.py"
      ;;
    stage2)
      runner="${PROJECT_ROOT}/scripts/run_stage2_with_overrides.py"
      ;;
    *)
      echo "Unsupported stage: ${stage}" >&2
      return 1
      ;;
  esac

  local cmd=(
    "${PYTHON_BIN}"
    "${runner}"
    --cfg-var-file "${cfg_var_file}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  echo
  echo "================================================================================"
  echo "[run_multi_stages] stage=${stage}"
  echo "[run_multi_stages] cfg_var_file=${cfg_var_file}"
  echo "[run_multi_stages] command=${cmd[*]}"
  echo "================================================================================"
  "${cmd[@]}"
}

cd "${PROJECT_ROOT}"

case "${MODE}" in
  stage1)
    run_stage "stage1" "${STAGE1_CFG}"
    ;;
  stage2)
    run_stage "stage2" "${STAGE2_CFG}"
    ;;
  both)
    run_stage "stage1" "${STAGE1_CFG}"
    run_stage "stage2" "${STAGE2_CFG}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    echo "Use one of: stage1, stage2, both" >&2
    exit 1
    ;;
esac
