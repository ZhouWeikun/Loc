#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="/home/data/zwk/pyproj_neuloc_v0"
TRAINER_PY="$REPO_ROOT/trainers/stage3_proxy_linearProjector_wANCE_evotorch.py"
VISLOC_YAML="$REPO_ROOT/trainer_depends/configs/stage3_visloc.yaml"
WINGTRA_YAML="$REPO_ROOT/trainer_depends/configs/stage3_wingtra.yaml"
STAGE3_RECALL_CFG="${STAGE3_RECALL_CFG:-per_scene}"
STAGE3_RECALL_CFG_YAML="${STAGE3_RECALL_CFG_YAML:-$REPO_ROOT/trainer_depends/configs/stage3_recall_thresholds.yaml}"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="neuloc_wisp"
CONDA_LIB="/root/miniconda3/envs/${CONDA_ENV}/lib"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[FATAL] missing conda setup script: $CONDA_SH" >&2
  exit 1
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"

run_one() {
  local base_yaml="$1"
  local selected_scene="$2"
  local ckpt_path="$3"
  local opts_path="$4"
  local recall_cfg="${5:-$STAGE3_RECALL_CFG}"

  local ckpt_dir
  local ckpt_file
  local ckpt_stem
  local exp_name
  local output_root
  local log_path
  local status_path

  ckpt_dir="$(dirname "$ckpt_path")"
  ckpt_file="$(basename "$ckpt_path")"
  ckpt_stem="${ckpt_file%.pth}"
  exp_name="stage3_test_$(basename "$ckpt_dir")__${ckpt_stem}"
  output_root="${ckpt_dir}/res_${ckpt_stem}"
  log_path="${output_root}/stage3_test_stdout.log"
  status_path="${output_root}/stage3_test_status.txt"

  mkdir -p "$output_root"

  {
    echo "[START] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "base_yaml=$base_yaml"
    echo "selected_scene_name=$selected_scene"
    echo "load_stage2_ckpt=$ckpt_path"
    echo "inherit_stage2_yaml=$opts_path"
    echo "stage3_analysis_export_root=$output_root"
    echo "stage3_recall_cfg=$recall_cfg"
    echo "stage3_recall_cfg_yaml=$STAGE3_RECALL_CFG_YAML"
    echo "exp_name=$exp_name"
  } | tee "$status_path"

  python "$TRAINER_PY" \
    --p_yaml "$base_yaml" \
    --selected_scene_name "$selected_scene" \
    --exp_name "$exp_name" \
    --load_stage2_ckpt "$ckpt_path" \
    --inherit_stage2_yaml "$opts_path" \
    --stage3_analysis_export_root "$output_root" \
    --stage3_recall_cfg "$recall_cfg" \
    --stage3_recall_cfg_yaml "$STAGE3_RECALL_CFG_YAML" \
    2>&1 | tee "$log_path"

  local rc=${PIPESTATUS[0]}
  {
    echo "[END] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "exit_code=$rc"
  } | tee -a "$status_path"

  if [[ $rc -ne 0 ]]; then
    echo "[ERROR] failed: $ckpt_path" >&2
    return "$rc"
  fi
  return 0
}

#declare -a RUN_SPECS=(
#  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch990_R1=88_MidErr=73m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch830_R1=84_MidErr=72m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_04|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch750_R1=86_MidErr=103m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_04|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch780_R1=86_MidErr=102m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_04|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch880_R1=81_MidErr=104m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch970_R1=73_MidErr=28m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch730_R1=61_MidErr=36m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch960_R1=83_MidErr=25m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie_1/epoch970_R1=75_MidErr=29m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie_1/opts.yaml"
#)

# Last-epoch wRejectSampling Stage-2 checkpoints.
#declare -a RUN_SPECS=(
#  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch990_R1=88_MidErr=73m.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_04|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$VISLOC_YAML|visloc_04|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc04_segment82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie_1/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie_1/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie_1/epoch990.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie_1/opts.yaml"
#  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_netvlad_codebookW18_mlpH1024B2_PN1cubie/epoch990.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_segment91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_netvlad_codebookW18_mlpH1024B2_PN1cubie/opts.yaml"
#)

# Usage examples:
#   bash scripts/run_stage3_test_best_stage2_wreject.sh
#   Each RUN_SPECS item can optionally append a 5th field: recall_cfg
#
# gem vs salad paired Stage-2 checkpoints.
# Format:
#   base_yaml|selected_scene|ckpt_path|opts_path|recall_cfg
declare -a RUN_SPECS=(
  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie_1/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie_1/opts.yaml|per_scene"
  "$WINGTRA_YAML|zurich|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml|per_scene"
  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW18_mlpH1024B2_PN1cubie/opts.yaml|per_scene"
  "$WINGTRA_YAML|zuchwil|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_zuchwil_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/opts.yaml|per_scene"
  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW19_mlpH1024B1_PN1cubie_1/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_gem_codebookW19_mlpH1024B1_PN1cubie_1/opts.yaml|per_scene"
  "$VISLOC_YAML|visloc_03|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B1_PN1cubie_1/epoch999.pth|$REPO_ROOT/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B1_PN1cubie_1/opts.yaml|per_scene"
)



total="${#RUN_SPECS[@]}"
for idx in "${!RUN_SPECS[@]}"; do
  IFS='|' read -r base_yaml selected_scene ckpt_path opts_path recall_cfg_override <<< "${RUN_SPECS[$idx]}"
  recall_cfg_override="${recall_cfg_override:-$STAGE3_RECALL_CFG}"
  echo "[BATCH] $((idx + 1))/$total -> $(basename "$(dirname "$ckpt_path")") / $(basename "$ckpt_path") / recall_cfg=${recall_cfg_override}"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[FATAL] missing ckpt: $ckpt_path" >&2
    exit 1
  fi
  if [[ ! -f "$opts_path" ]]; then
    echo "[FATAL] missing opts: $opts_path" >&2
    exit 1
  fi
  run_one "$base_yaml" "$selected_scene" "$ckpt_path" "$opts_path" "$recall_cfg_override" || exit $?
done

echo "[DONE] all stage3 best-stage2 test jobs completed."
