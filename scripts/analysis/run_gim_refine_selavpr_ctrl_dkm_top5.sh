#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gim
REPO_ROOT=/home/data/zwk/pyproj_neuloc_v0
MATCHING_REFINE_ROOT="${REPO_ROOT}/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/mathing_refine"

SUMMARY_CSV="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap000/stage1_interval_gallery_overlap000.csv"
python "${REPO_ROOT}/scripts/analysis/retrieval_then_matching/retrieval_then_gim_refine.py" \
  --summary-csv "${SUMMARY_CSV}" \
  --aggregator selavpr \
  --matcher-model gim_dkm \
  --topn-match 5 \
  --save-intermediates \
  --output-root "${MATCHING_REFINE_ROOT}/selavpr_ctrl_dkm_top5" \
  --summary-out "${MATCHING_REFINE_ROOT}/selavpr_ctrl_dkm_top5_summary.csv"

SUMMARY_CSV="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_segment_netvlad&salad&selavpr_gallery_overlap000/stage1_segment_gallery_overlap000.csv"
python "${REPO_ROOT}/scripts/analysis/retrieval_then_matching/retrieval_then_gim_refine.py" \
  --summary-csv "${SUMMARY_CSV}" \
  --aggregator selavpr \
  --matcher-model gim_dkm \
  --topn-match 5 \
  --save-intermediates \
  --output-root "${MATCHING_REFINE_ROOT}/selavpr_ctrl_dkm_top5" \
  --summary-out "${MATCHING_REFINE_ROOT}/selavpr_ctrl_dkm_top5_summary.csv"