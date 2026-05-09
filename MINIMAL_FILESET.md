# Continuloc Minimal V1 File Set

This branch keeps the smallest complete code path for the three-stage INGP
pipeline. The anchor entrypoints are:

- `trainers/stage1_visual_encoder_minimal.py`
- `trainers/stage2_INGP_minimal.py`
- `trainers/stage3_ingp_inferencer_minimal.py`

## Commands

Stage 1 train/test:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_wingtra.yaml
```

Stage 2 train/test:

```bash
python trainers/stage2_INGP_minimal.py \
  --p_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml
```

Stage 3 INGP inference:

```bash
python trainers/stage3_ingp_inferencer_minimal.py \
  --stage3_yaml trainer_depends/configs/stage3_ingp_minimal_wingtra.yaml \
  --stage2_opts_yaml <path-to-stage2-opts.yaml>
```

Use the matching `*_visloc.yaml` files for the VisLoc profile.

## Kept Surface

- Visual backbone: DINOv2 and DINOv3 only.
- Aggregator: SALAD residual only.
- Stage 2 field: INGP hash grid plus residual conditional grid MLP.
- Stage 3: seed-mode coarse search, mode packing, CMA refinement, and
  progressive Recall@K reporting.
- Configs: Wingtra and VisLoc profiles, stage3 recall thresholds, and the two
  hash-grid configs used by those profiles.

## Removed Surface

The branch intentionally removes legacy trainers, ANCE/gallery-bank helpers,
APR/MetricNet/proxy Stage3 paths, retrieval-then-matching scripts, batch
analysis/export/visualization scripts, non-main model options, IDE metadata,
and Python cache artifacts.

## External Inputs

Dataset JSON/CSV paths, pretrained DINO weights, Kaolin Wisp, and trained
Stage1/Stage2 checkpoints remain external to this repository. The YAML files
record the expected path shape, and command-line overrides should be used when
running on another machine.
