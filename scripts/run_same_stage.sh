# segment / overlap000
#python /home/data/zwk/pyproj_neuloc_v0/scripts/analysis/run_stage1_gallery_fm_plan.py \
#    --plan-yaml "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_segment_netvlad&salad&selavpr_gallery_overlap000/stage1_segment_eval_plan_overlap000.yaml" \
#    --output-csv "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_segment_netvlad&salad&selavpr_gallery_overlap000/stage1_segment_gallery_overlap000.csv" \
#    --scale-ratio-th 1.2 \
#    --skip-existing
#
## interval / overlap000
#python /home/data/zwk/pyproj_neuloc_v0/scripts/analysis/run_stage1_gallery_fm_plan.py \
#    --plan-yaml "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap000/stage1_interval_eval_plan_overlap000.yaml" \
#    --output-csv "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap000/stage1_interval_gallery_overlap000.csv" \
#    --scale-ratio-th 1.2 \
#    --skip-existing

#  1. interval / overlap025
#python /home/data/zwk/pyproj_neuloc_v0/scripts/analysis/run_stage1_gallery_fm_plan.py \
#    --plan-yaml "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap025/stage1_interval_eval_plan_overlap025.yaml" \
#    --output-csv "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_interval_netvlad&salad&selavpr_gallery_overlap025/stage1_interval_gallery_overlap025.csv" \
#    --scale-ratio-th 1.2 \
#    --skip-existing

##  2. segment / overlap050
#python /home/data/zwk/pyproj_neuloc_v0/scripts/analysis/run_stage1_gallery_fm_plan.py \
#    --plan-yaml "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_segment_netvlad&salad&selavpr_gallery_overlap050/stage1_segment_eval_plan_overlap050.yaml" \
#    --output-csv "/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/analysis/stage1_crtl_ckpts2exps/exps_segment_netvlad&salad&selavpr_gallery_overlap050/stage1_segment_gallery_overlap050.csv" \
#    --scale-ratio-th 1.2 \
#    --skip-existing

## stage2:
python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_with_overrides.py --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage2_cfg_wingtra.txt --base-yaml trainer_depends/configs/stage2_INGP_wingtra.yaml --grid-base-yaml trainer_depends/configs/nerf_hash_wingtra.yaml
##
#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_with_overrides.py --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage2_cfg_visloc.txt --base-yaml trainer_depends/configs/stage2_INGP_visloc.yaml --grid-base-yaml trainer_depends/configs/nerf_hash_visloc.yaml

#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_with_overrides.py --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage2_cfg_wingtra_tmp.txt --base-yaml trainer_depends/configs/stage2_INGP_wingtra.yaml --grid-base-yaml trainer_depends/configs/nerf_hash_wingtra.yaml
#
#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_with_overrides.py --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage2_cfg_visloc_tmp.txt --base-yaml trainer_depends/configs/stage2_INGP_visloc.yaml --grid-base-yaml trainer_depends/configs/nerf_hash_visloc.yaml



#stage1,ctrl 实验：
#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#  --train-script /home/data/zwk/pyproj_neuloc_v0/trainers/stage1_visual_encoder_controlexps.py \
#  --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_visloc.yaml \
#  --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_visloc_ctrl.txt

#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#  --train-script /home/data/zwk/pyproj_neuloc_v0/trainers/stage1_visual_encoder_controlexps.py \
#  --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_wingtra.yaml \
#  --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_wingtra_ctrl.txt \


#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#  --train-script /home/data/zwk/pyproj_neuloc_v0/trainers/stage1_visual_encoder_controlexps.py \
#  --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_visloc_ctrl_wRandSatNeg.txt \
#  --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_visloc.yaml
#
#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#--train-script /home/data/zwk/pyproj_neuloc_v0/trainers/stage1_visual_encoder_controlexps.py \
#--cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_wingtra_ctrl_wRandSatNeg.txt \
#--base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_wingtra.yaml

  #stage1,ours 实验：
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#  --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_wingtra.txt \
#  --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_wingtra.yaml
#python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py \
#  --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_visloc.txt \
#  --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_visloc.yaml

# APR,interval:
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=zurich \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_interval91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad_1/epoch004_zurichR1=82MedErr23m_zuchwilR1=76MedErr=25m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_interval91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad_1/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_zurich_interval91
#
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=zuchwil \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_interval91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad_1/epoch004_zurichR1=82MedErr23m_zuchwilR1=76MedErr=25m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_interval91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad_1/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_zuchwil_interval91

#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=visloc_03 \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_interval82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch035_03R1=94MeanErr101mMedErr73m_04R1=93MeanErr161MedErr99m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_interval82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_visloc03_interval82 \
#  --set network_setting.apr_mlp_num_layers=6
#
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=visloc_04 \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_interval82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch035_03R1=94MeanErr101mMedErr73m_04R1=93MeanErr161MedErr99m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_interval82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_visloc04_interval82 \
#  --set network_setting.apr_mlp_num_layers=6
#
## APR,segment:
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=zurich \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_segment91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch007_zurichR1=80MidErr24m_zuchwilR1=70MidErr28m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_segment91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_zurich_segment91
#
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=zuchwil \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_segment91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch007_zurichR1=80MidErr24m_zuchwilR1=70MidErr28m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_segment91_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_zuchwil_segment91
#
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=visloc_03 \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_segment82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch014_03R85MidErr66m_04R90MidErr90m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_segment82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_visloc03_segment82 \
#  --set network_setting.apr_mlp_num_layers=6
#
#/root/miniconda3/envs/neuloc_wisp/bin/python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage2_arp_mlp.py\
#  --set scenes_setting.selected_scene_name=visloc_04 \
#  --set exp_setting.load_stage1_ckpt=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_segment82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/epoch014_03R85MidErr66m_04R90MidErr90m.pth \
#  --set exp_setting.inherit_stage1_yaml=/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_visloc_segment82_wRejectSampling_tripleLossSingleEdgeHardestFmMask_dinov2_adF4_salad/opts.yaml \
#  --set exp_setting.exp_name=stage2_apr_mlp_visloc04_segment82 \
#  --set network_setting.apr_mlp_num_layers=6

