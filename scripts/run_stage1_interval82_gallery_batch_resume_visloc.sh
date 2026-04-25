#!/usr/bin/env bash
set -eo pipefail

cd /home/data/zwk/pyproj_neuloc_v0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuloc_wisp

echo "[ResumeBatch] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

specs=(
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem/epoch049.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem|visloc_03"
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem/epoch049.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_gem|visloc_04"
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad/epoch049.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad|visloc_03"
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad/epoch049.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_netvlad|visloc_04"
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad/epoch060.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad|visloc_03"
  "gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad/opts.yaml|gen_fm_exps/ckpts/stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad/epoch060.pth|stage1_visloc_interval82_wRandSatNeg_msloss_dinov2B2_salad|visloc_04"
)

total=${#specs[@]}
idx=0
for spec in "${specs[@]}"; do
  idx=$((idx + 1))
  IFS="|" read -r py ckpt exp scene <<< "$spec"
  echo "[ResumeBatch] ($idx/$total) start scene=$scene exp=$exp ckpt=$ckpt at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python trainers/stage1_visual_encoder_w_ANCE.py \
    --test_only true \
    --test_mode gallery_bank \
    --p_yaml "$py" \
    --load2test "$ckpt" \
    --exp_name_override "$exp" \
    --scene_name "$scene"
  status=$?
  echo "[ResumeBatch] ($idx/$total) end status=$status scene=$scene exp=$exp at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "[ResumeBatch] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
