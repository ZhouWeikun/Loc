#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified wrapper for Stage-1 / Stage-2 override launchers.

Examples:
  python scripts/run_stage_with_overrides.py \
    --stage stage1 \
    --cfg-var-file gen_fm_exps/run_yamls/stage1_cfg_var.txt

  python scripts/run_stage_with_overrides.py \
    --stage stage2 \
    --cfg-var-file gen_fm_exps/run_yamls/stage2_cfg_var.txt

  python scripts/run_stage_with_overrides.py \
    --stage stage2 \
    --set exp_setting.exp_name=stage2_visloc_debug \
    --set gridcfg.codebook_bitwidth=20 \
    --dry-run

Python usage:
  from scripts.run_stage_with_overrides import run_stage_with_overrides

  run_stage_with_overrides(
      stage="stage1",
      cfg_var_file="gen_fm_exps/run_yamls/stage1_cfg_var.txt",
      dry_run=True,
  )
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from scripts.run_stage1_with_overrides import run_stage1_with_overrides
from scripts.run_stage2_with_overrides import run_stage2_with_overrides


def run_stage_with_overrides(
    stage,
    base_yaml=None,
    output_dir=None,
    train_script=None,
    python_bin=sys.executable,
    overrides=None,
    cfg_var_file=None,
    dry_run=False,
    grid_base_yaml=None,
):
    """
    Unified callable entry for Stage-1 / Stage-2 override launchers.

    Args:
        stage: "stage1" or "stage2".
        base_yaml: Optional stage-specific base YAML path.
        output_dir: Optional directory for generated YAMLs.
        train_script: Optional stage-specific training entry path.
        python_bin: Python executable used to launch training.
        overrides: Override payload accepted by the underlying stage launcher.
        cfg_var_file: Optional batch experiment file path.
        dry_run: If True, only generate YAMLs and print commands.
        grid_base_yaml: Stage-2 only. Optional base grid YAML.

    Returns:
        A list of result dicts from the underlying stage launcher.
    """
    stage = str(stage).strip().lower()
    if stage == "stage1":
        kwargs = {
            "python_bin": python_bin,
            "overrides": overrides,
            "cfg_var_file": cfg_var_file,
            "dry_run": dry_run,
        }
        if base_yaml is not None:
            kwargs["base_yaml"] = base_yaml
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
        if train_script is not None:
            kwargs["train_script"] = train_script
        return run_stage1_with_overrides(**kwargs)

    if stage == "stage2":
        kwargs = {
            "python_bin": python_bin,
            "overrides": overrides,
            "cfg_var_file": cfg_var_file,
            "dry_run": dry_run,
        }
        if base_yaml is not None:
            kwargs["base_yaml"] = base_yaml
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
        if train_script is not None:
            kwargs["train_script"] = train_script
        if grid_base_yaml is not None:
            kwargs["grid_base_yaml"] = grid_base_yaml
        return run_stage2_with_overrides(**kwargs)

    raise ValueError(f"Unsupported stage: {stage!r}. Expected 'stage1' or 'stage2'.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified wrapper for scripts/run_stage1_with_overrides.py and scripts/run_stage2_with_overrides.py"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("stage1", "stage2"),
        help="Which stage launcher to use.",
    )
    parser.add_argument(
        "--base-yaml",
        default=None,
        help="Optional base YAML path. Defaults to the selected stage launcher default.",
    )
    parser.add_argument(
        "--grid-base-yaml",
        default=None,
        help="Stage-2 only. Optional base nerf_hash YAML for temporary grid configs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory used to store generated YAML files.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Optional training entry script path.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch training.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Override in the form section.key=value. "
            "For Stage-2, grid YAML overrides can use gridcfg.<key>=value."
        ),
    )
    parser.add_argument(
        "--cfg-var-file",
        default=None,
        help="Optional batch experiment file path. Defaults to the selected stage launcher behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate YAMLs and print commands, without launching training.",
    )
    args = parser.parse_args()

    run_stage_with_overrides(
        stage=args.stage,
        base_yaml=args.base_yaml,
        grid_base_yaml=args.grid_base_yaml,
        output_dir=args.output_dir,
        train_script=args.train_script,
        python_bin=args.python_bin,
        overrides=args.overrides,
        cfg_var_file=args.cfg_var_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
