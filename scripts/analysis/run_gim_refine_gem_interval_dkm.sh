#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gim
REPO_ROOT=/home/data/zwk/pyproj_neuloc_v0
MATCHING_REFINE_ROOT="${REPO_ROOT}/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/mathing_refine"

python "${REPO_ROOT}/scripts/analysis/retrieval_then_matching/retrieval_then_gim_refine.py" \
  --bundle-path "${REPO_ROOT}/gen_fm_exps/gallery_bank_stage1/visloc_03_overlap050_bins46x57x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt" \
  --bundle-path "${REPO_ROOT}/gen_fm_exps/gallery_bank_stage1/visloc_04_overlap050_bins56x21x36x4_linear/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt" \
  --bundle-path "${REPO_ROOT}/gen_fm_exps/gallery_bank_stage1/zuchwil_overlap050_bins48x67x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt" \
  --bundle-path "${REPO_ROOT}/gen_fm_exps/gallery_bank_stage1/zurich_overlap050_bins63x45x36x4_linear/stage1_wingtra_interval82_wRandSatNeg_msloss_dinov2B2_gem_epoch049/stage1_retrieval_eval_bundle.pt" \
  --matcher-model gim_dkm \
  --topn-match 10 \
  --output-root "${MATCHING_REFINE_ROOT}/gem_interval_dkm" \
  --summary-out "${MATCHING_REFINE_ROOT}/gem_interval_dkm_summary.csv"
