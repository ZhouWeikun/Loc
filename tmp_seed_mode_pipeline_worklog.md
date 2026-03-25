# Seed-Mode Pipeline Worklog

Date: 2026-03-13
Target environment: `neuloc_wisp`

## This round

- Added a new pipeline module:
  - `trainers/util_stage3_multi_stage_refiner.py`
- Implemented the first runnable components:
  - `TopNSeedScreening`
  - `LocalSeedCloudBuilder`
  - `IterativeSeedCloudRefiner`
  - `LightweightModeDeduper`
  - `SobolBoxModeSampler`
  - `DiagonalEliteModeRefiner`
  - `RatioModePruner`
  - `EvoTorchFinalModeOptimizer`
- Added a new evaluation entry in:
  - `trainers/stage3_bpf_proxy_linearProjector_wANCE_evotorch.py`
  - method: `_test_3d_fine_accuracy_seed_mode_CMA_ES(...)`

## Current design choices

- Stage 2 uses:
  - diagonal sigma only
  - `selection_metric` for pruning
  - `best_score` for final output
- Stage 1 now supports:
  - iterative seed-cloud relocate / resample / prune
  - per-round sample-count schedule
  - per-round radius-scale schedule
  - optional fixed-scale resampling
- Stage 3 CMA init sigma uses:
  - Stage 2 mode sigma first
  - fallback = Stage 1 xy bin-size mean
- The existing CMA path is kept intact.
- The new seed-mode pipeline path is added in parallel, not yet made default.

## Known limitations

- Stage 2 currently samples in a box `center +/- sigma_diag_raw`, not a full Gaussian.
- Stage 2 covariance is diagonal only.
- Stage 3 currently runs one CMA search per surviving mode to preserve per-mode sigma.
- The new path is implemented, but not yet wired as the default in `test()`.

## Next likely steps

- Compare old CMA path vs new seed-mode path on a small query batch.
- Decide whether Stage 2 sampling should stay box-based or move to diagonal Gaussian.
- If the new path is stable, add a switch in `test()` to choose between:
  - old direct-CMA path
  - new seed-mode pipeline path
