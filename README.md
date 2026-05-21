# Continuloc Minimal INGP Pipeline

This repository contains the minimal three-stage Continuloc pipeline for UAV visual localization:

| Stage | Purpose | Entry point |
| --- | --- | --- |
| Stage 1 | Train and evaluate the visual descriptor model | `trainers/stage1_visual_encoder_minimal.py` |
| Stage 2 | Train and evaluate the INGP coordinate descriptor field | `trainers/stage2_INGP_minimal.py` |
| Stage 3 | Run final INGP localization inference and Recall evaluation | `trainers/stage3_ingp_inferencer_minimal.py` |

Stage 1 and Stage 2 support both training and testing. Stage 3 is inference-only.

## Setup

Run the commands below from the repository root.

The training code expects a PyTorch environment with the data and model dependencies already installed. Stage 2 and Stage 3 also require Kaolin Wisp for the INGP hash grid. If Wisp is not in the default lookup path, set:

```bash
export KAOLIN_WISP_ROOT=/path/to/kaolin-wisp
```

The visual backbone weights must be available locally. DINOv2 uses its configured `pretrained_path` or `weights_dir`; DINOv3 uses `backbone_config.weights_path`.

The dataset release is in progress and will be made available soon.

## Configuration

The pipeline is configured with YAML files:

- Wingtra:
  - `trainer_depends/configs/stage1_visual_encoder_wingtra.yaml`
  - `trainer_depends/configs/stage2_INGP_wingtra.yaml`
- VisLoc:
  - `trainer_depends/configs/stage1_visual_encoder_visloc.yaml`
  - `trainer_depends/configs/stage2_INGP_visloc.yaml`
- Stage 3:
  - `trainer_depends/configs/stage3_ingp_minimal.yaml`

Before running, update the YAML paths for your machine. Each scene needs:

```yaml
scenes_setting:
  dataset_name: wingtra
  scenes:
    - name: zurich
      p_satinfo_json: /path/to/satellite_info.json
      p_uavinfo_json: /path/to/uav_metadata.json
      p_uav_geocsv: /path/to/uav_geo.csv
```

After the first checkpoint is saved, each training run writes an `opts.yaml` into its checkpoint directory. Reuse that file for later testing and for the next stage when possible.

Two fields are especially important:

- Stage 1 minimal accepts `loss_type: triplet_loss` or `loss_type: infonce`.
- In Stage 2, `load_stage1_ckpt` points to a Stage 1 `.pth` checkpoint, while `inherit_stage1_yaml` points to the matching Stage 1 `opts.yaml`.

## Stage 1: Visual Descriptor Model

Train Stage 1:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_wingtra.yaml
```

Test a trained Stage 1 run:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml /path/to/stage1_run/opts.yaml \
  --load2test /path/to/stage1_run/epoch099.pth \
  --test_only
```

Stage 1 evaluation builds a satellite gallery and reports Recall@K for the configured scenes.

## Stage 2: INGP Descriptor Field

Stage 2 uses a trained Stage 1 model. Set the Stage 1 checkpoint and config in the Stage 2 YAML or pass them on the command line:

```bash
python trainers/stage2_INGP_minimal.py \
  --p_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml \
  --load_stage1_ckpt /path/to/stage1_run/epoch099.pth \
  --inherit_stage1_yaml /path/to/stage1_run/opts.yaml
```

Test a trained Stage 2 run:

```bash
python trainers/stage2_INGP_minimal.py \
  --p_yaml /path/to/stage2_run/opts.yaml \
  --load2test /path/to/stage2_run/epoch999.pth \
  --test_only
```

Stage 2 evaluation uses UAV image descriptors as queries and Stage 2 coordinate descriptors as the gallery.

## Stage 3: Final Localization Evaluation

Run Stage 3 from a Stage 2 experiment:

```bash
python trainers/stage3_ingp_inferencer_minimal.py \
  --stage3_yaml trainer_depends/configs/stage3_ingp_minimal.yaml \
  --stage2_opts_yaml /path/to/stage2_run/opts.yaml
```

If the Stage 2 checkpoint directory contains `epoch*.pth`, the script selects the checkpoint with the largest epoch index. A checkpoint can also be provided explicitly:

```bash
python trainers/stage3_ingp_inferencer_minimal.py \
  --stage3_yaml trainer_depends/configs/stage3_ingp_minimal.yaml \
  --stage2_opts_yaml /path/to/stage2_run/opts.yaml \
  --load_stage2_ckpt /path/to/stage2_run/epoch999.pth \
  --load_stage1_ckpt /path/to/stage1_run/epoch099.pth \
  --stage3_print_progressive_recall true
```

Stage 3 performs coarse search, mode refinement, CMA-ES refinement, and progressive Recall evaluation.

## End-to-End Example

```bash
# Stage 1
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_wingtra.yaml

# Stage 2
python trainers/stage2_INGP_minimal.py \
  --p_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml \
  --load_stage1_ckpt /path/to/stage1_run/epoch099.pth \
  --inherit_stage1_yaml /path/to/stage1_run/opts.yaml

# Stage 3
python trainers/stage3_ingp_inferencer_minimal.py \
  --stage3_yaml trainer_depends/configs/stage3_ingp_minimal.yaml \
  --stage2_opts_yaml /path/to/stage2_run/opts.yaml \
  --stage3_print_progressive_recall true
```
