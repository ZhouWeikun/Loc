#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gim
REPO_ROOT=/home/data/zwk/pyproj_neuloc_v0
MATCHING_REFINE_ROOT="${REPO_ROOT}/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/mathing_refine"
SUMMARY_CSV="${REPO_ROOT}/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap050/stage1_ctrl_gallery_overlap050.csv"

python "${REPO_ROOT}/scripts/analysis/retrieval_then_matching/retrieval_then_gim_refine.py" \
  --summary-csv "${SUMMARY_CSV}" \
  --aggregator selavpr \
  --matcher-model gim_roma \
  --topn-match 5 \
  --save-intermediates \
  --output-root "${MATCHING_REFINE_ROOT}/selavpr_ctrl_roma_top5" \
  --summary-out "${MATCHING_REFINE_ROOT}/selavpr_ctrl_roma_top5_summary.csv"
