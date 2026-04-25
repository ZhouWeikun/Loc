#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sequential Stage-1 / Stage-2 pipeline launcher.

This script is meant to be a single entry point when you want to queue
multiple stages before leaving the machine unattended.

Examples:
  python scripts/run_stage_pipeline_with_overrides.py \
    --sequence stage1,stage2

  python scripts/run_stage_pipeline_with_overrides.py \
    --sequence stage1,stage2 \
    --stage1-cfg-var-file gen_fm_exps/run_yamls/stage1_cfg_var.txt \
    --stage2-cfg-var-file gen_fm_exps/run_yamls/stage2_cfg_var.txt

  python scripts/run_stage_pipeline_with_overrides.py \
    --sequence stage2,stage1 \
    --stage2-set exp_setting.exp_name=stage2_debug \
    --stage1-set exp_setting.exp_name=stage1_debug \
    --dry-run

Python usage:
  from scripts.run_stage_pipeline_with_overrides import run_stage_pipeline

  run_stage_pipeline(
      sequence=("stage1", "stage2"),
      stage1_cfg_var_file="gen_fm_exps/run_yamls/stage1_cfg_var.txt",
      stage2_cfg_var_file="gen_fm_exps/run_yamls/stage2_cfg_var.txt",
      dry_run=True,
  )
"""

import argparse
import os
import re
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from scripts.run_stage1_with_overrides import (
    DEFAULT_BASE_YAML as STAGE1_DEFAULT_BASE_YAML,
    DEFAULT_CFG_VAR_FILE as STAGE1_DEFAULT_CFG_VAR_FILE,
    DEFAULT_TMP_DIR as DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_SCRIPT as STAGE1_DEFAULT_TRAIN_SCRIPT,
    _load_cfg_var_experiments,
    _load_yaml,
    _normalize_override_pairs,
    run_stage1_with_overrides,
)
from scripts.run_stage2_with_overrides import (
    DEFAULT_BASE_YAML as STAGE2_DEFAULT_BASE_YAML,
    DEFAULT_CFG_VAR_FILE as STAGE2_DEFAULT_CFG_VAR_FILE,
    DEFAULT_GRID_BASE_YAML as STAGE2_DEFAULT_GRID_BASE_YAML,
    DEFAULT_TRAIN_SCRIPT as STAGE2_DEFAULT_TRAIN_SCRIPT,
    run_stage2_with_overrides,
)


def _resolve_path(path_text):
    if path_text is None:
        return None
    if path_text == "":
        return ""
    if os.path.isabs(str(path_text)):
        return str(path_text)
    return os.path.abspath(os.path.join(PROJECT_ROOT, str(path_text)))


def _parse_sequence(sequence):
    if isinstance(sequence, (list, tuple)):
        stages = [str(item).strip().lower() for item in sequence if str(item).strip()]
    else:
        stages = [item.strip().lower() for item in str(sequence).split(",") if item.strip()]
    if not stages:
        raise ValueError("Empty sequence. Expected something like 'stage1,stage2'.")
    for stage in stages:
        if stage not in ("stage1", "stage2"):
            raise ValueError(f"Unsupported stage in sequence: {stage!r}")
    return stages


def _load_single_experiment_override_pairs(base_yaml, cfg_var_file, cli_overrides, experiment_index=1):
    override_pairs = []
    cli_override_pairs = _normalize_override_pairs(cli_overrides)

    if cfg_var_file:
        base_config = _load_yaml(_resolve_path(base_yaml))
        cfg_path = _resolve_path(cfg_var_file)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"cfg_var file not found: {cfg_path}")
        experiments = _load_cfg_var_experiments(cfg_path, base_config)
        if len(experiments) > 0:
            experiment_index = int(experiment_index)
            if experiment_index < 1 or experiment_index > len(experiments):
                raise ValueError(
                    f"experiment_index out of range for {cfg_path}: "
                    f"{experiment_index} not in [1, {len(experiments)}]"
                )
            override_pairs.extend(experiments[experiment_index - 1][1])

    override_pairs.extend(cli_override_pairs)
    return override_pairs


def _find_latest_epoch_ckpt(directory):
    if not directory or not os.path.isdir(directory):
        return None

    best = None
    for name in os.listdir(directory):
        match = re.fullmatch(r"epoch(\d+)(?:.*)?\.pth", name)
        if not match:
            continue
        epoch = int(match.group(1))
        path = os.path.join(directory, name)
        score = (epoch, int("best" in name), os.path.getmtime(path))
        if best is None or score > best[0]:
            best = (score, path)
    return None if best is None else best[1]


def _resolve_run_dir(root_dir, requested_exp_name, started_at):
    root_dir = _resolve_path(root_dir)
    if not root_dir or not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        if name == requested_exp_name or name.startswith(f"{requested_exp_name}_"):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path):
                candidates.append(path)
    if not candidates:
        return None

    recent = [path for path in candidates if os.path.getmtime(path) >= started_at - 2.0]
    pool = recent if recent else candidates
    return max(pool, key=os.path.getmtime)


def _collect_run_artifacts(generated_yaml, started_at):
    run_config = _load_yaml(generated_yaml)
    exp_setting = run_config.get("exp_setting") or {}
    requested_exp_name = str(exp_setting.get("exp_name", ""))
    ckpt_root = exp_setting.get("dir2save_ckpt", "gen_fm_exps/ckpts")
    log_root = exp_setting.get("dir2save_log", "gen_fm_exps/logs")

    ckpt_dir = _resolve_run_dir(ckpt_root, requested_exp_name, started_at)
    log_dir = _resolve_run_dir(log_root, requested_exp_name, started_at)
    latest_checkpoint = _find_latest_epoch_ckpt(ckpt_dir)

    return {
        "requested_exp_name": requested_exp_name,
        "ckpt_root": _resolve_path(ckpt_root),
        "log_root": _resolve_path(log_root),
        "ckpt_dir": ckpt_dir,
        "log_dir": log_dir,
        "latest_checkpoint": latest_checkpoint,
    }


def _run_stage1_single(
    *,
    base_yaml,
    output_dir,
    train_script,
    python_bin,
    cfg_var_file,
    cfg_experiment_index,
    cli_overrides,
    dry_run,
):
    override_pairs = _load_single_experiment_override_pairs(
        base_yaml, cfg_var_file, cli_overrides, experiment_index=cfg_experiment_index
    )
    started_at = time.time()
    results = run_stage1_with_overrides(
        base_yaml=base_yaml,
        output_dir=output_dir,
        train_script=train_script,
        python_bin=python_bin,
        overrides=override_pairs,
        cfg_var_file=None,
        dry_run=dry_run,
    )
    result = dict(results[0])
    result["stage"] = "stage1"
    if not dry_run:
        result.update(_collect_run_artifacts(result["generated_yaml"], started_at))
    return result


def _run_stage2_single(
    *,
    base_yaml,
    output_dir,
    train_script,
    python_bin,
    cfg_var_file,
    cfg_experiment_index,
    cli_overrides,
    dry_run,
    grid_base_yaml,
    injected_override_pairs=None,
):
    override_pairs = _load_single_experiment_override_pairs(
        base_yaml, cfg_var_file, cli_overrides, experiment_index=cfg_experiment_index
    )
    if injected_override_pairs:
        override_pairs.extend(list(injected_override_pairs))

    started_at = time.time()
    results = run_stage2_with_overrides(
        base_yaml=base_yaml,
        output_dir=output_dir,
        train_script=train_script,
        python_bin=python_bin,
        overrides=override_pairs,
        cfg_var_file=None,
        dry_run=dry_run,
        grid_base_yaml=grid_base_yaml,
    )
    result = dict(results[0])
    result["stage"] = "stage2"
    if not dry_run:
        result.update(_collect_run_artifacts(result["generated_yaml"], started_at))
    return result


def run_stage_pipeline(
    *,
    sequence=("stage1", "stage2"),
    python_bin=sys.executable,
    dry_run=False,
    auto_link_stage1_ckpt=True,
    stage1_base_yaml=STAGE1_DEFAULT_BASE_YAML,
    stage1_output_dir=DEFAULT_OUTPUT_DIR,
    stage1_train_script=STAGE1_DEFAULT_TRAIN_SCRIPT,
    stage1_cfg_var_file=STAGE1_DEFAULT_CFG_VAR_FILE,
    stage1_cfg_experiment_index=1,
    stage1_overrides=None,
    stage2_base_yaml=STAGE2_DEFAULT_BASE_YAML,
    stage2_output_dir=DEFAULT_OUTPUT_DIR,
    stage2_train_script=STAGE2_DEFAULT_TRAIN_SCRIPT,
    stage2_cfg_var_file=STAGE2_DEFAULT_CFG_VAR_FILE,
    stage2_cfg_experiment_index=1,
    stage2_overrides=None,
    stage2_grid_base_yaml=None,
):
    """
    Run a sequential stage pipeline.

    Notes:
    - Pipeline mode supports only one experiment per stage.
    - When sequence contains stage1 before stage2, stage2 can automatically
      inherit stage1's latest checkpoint via exp_setting.load_stage1_ckpt.
    """
    stages = _parse_sequence(sequence)
    pipeline_results = []
    stage1_latest_ckpt = None

    for idx, stage in enumerate(stages, start=1):
        print(f"\n{'#' * 100}")
        print(f"[Pipeline] Step {idx}/{len(stages)} -> {stage}")
        print(f"{'#' * 100}")

        if stage == "stage1":
            result = _run_stage1_single(
                base_yaml=_resolve_path(stage1_base_yaml),
                output_dir=_resolve_path(stage1_output_dir),
                train_script=_resolve_path(stage1_train_script),
                python_bin=python_bin,
                cfg_var_file=stage1_cfg_var_file,
                cfg_experiment_index=stage1_cfg_experiment_index,
                cli_overrides=stage1_overrides,
                dry_run=dry_run,
            )
            if not dry_run:
                stage1_latest_ckpt = result.get("latest_checkpoint")
                print(f"[Pipeline] stage1 ckpt_dir: {result.get('ckpt_dir')}")
                print(f"[Pipeline] stage1 latest checkpoint: {stage1_latest_ckpt}")
            pipeline_results.append(result)
            continue

        injected_override_pairs = []
        if auto_link_stage1_ckpt and stage1_latest_ckpt:
            injected_override_pairs.append(("exp_setting.load_stage1_ckpt", stage1_latest_ckpt))
            print(f"[Pipeline] auto-link stage1 ckpt -> stage2: {stage1_latest_ckpt}")
        elif auto_link_stage1_ckpt and "stage1" in stages[:idx - 1] and not dry_run:
            raise RuntimeError("stage1 finished but no checkpoint was found to pass into stage2.")

        result = _run_stage2_single(
            base_yaml=_resolve_path(stage2_base_yaml),
            output_dir=_resolve_path(stage2_output_dir),
            train_script=_resolve_path(stage2_train_script),
            python_bin=python_bin,
            cfg_var_file=stage2_cfg_var_file,
            cfg_experiment_index=stage2_cfg_experiment_index,
            cli_overrides=stage2_overrides,
            dry_run=dry_run,
            grid_base_yaml=_resolve_path(stage2_grid_base_yaml) if stage2_grid_base_yaml else None,
            injected_override_pairs=injected_override_pairs,
        )
        if not dry_run:
            print(f"[Pipeline] stage2 ckpt_dir: {result.get('ckpt_dir')}")
            print(f"[Pipeline] stage2 latest checkpoint: {result.get('latest_checkpoint')}")
        pipeline_results.append(result)

    return pipeline_results


def main():
    parser = argparse.ArgumentParser(
        description="Run a sequential Stage-1 / Stage-2 pipeline from one entry point."
    )
    parser.add_argument(
        "--sequence",
        default="stage1,stage2",
        help="Comma-separated stage order, e.g. 'stage1,stage2' or 'stage2,stage1'.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch stage scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate YAMLs and print commands, without launching training.",
    )
    parser.add_argument(
        "--no-auto-link-stage1-ckpt",
        dest="auto_link_stage1_ckpt",
        action="store_false",
        help="Disable automatic injection of stage1 checkpoint into stage2.",
    )

    parser.add_argument("--stage1-base-yaml", default=STAGE1_DEFAULT_BASE_YAML)
    parser.add_argument("--stage1-output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-train-script", default=STAGE1_DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--stage1-cfg-var-file", default=STAGE1_DEFAULT_CFG_VAR_FILE)
    parser.add_argument("--stage1-cfg-index", type=int, default=1)
    parser.add_argument("--stage1-set", dest="stage1_overrides", action="append", default=[])

    parser.add_argument("--stage2-base-yaml", default=STAGE2_DEFAULT_BASE_YAML)
    parser.add_argument("--stage2-output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage2-train-script", default=STAGE2_DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--stage2-cfg-var-file", default=STAGE2_DEFAULT_CFG_VAR_FILE)
    parser.add_argument("--stage2-cfg-index", type=int, default=1)
    parser.add_argument("--stage2-grid-base-yaml", default=None)
    parser.add_argument("--stage2-set", dest="stage2_overrides", action="append", default=[])

    args = parser.parse_args()

    run_stage_pipeline(
        sequence=args.sequence,
        python_bin=args.python_bin,
        dry_run=args.dry_run,
        auto_link_stage1_ckpt=args.auto_link_stage1_ckpt,
        stage1_base_yaml=args.stage1_base_yaml,
        stage1_output_dir=args.stage1_output_dir,
        stage1_train_script=args.stage1_train_script,
        stage1_cfg_var_file=args.stage1_cfg_var_file,
        stage1_cfg_experiment_index=args.stage1_cfg_index,
        stage1_overrides=args.stage1_overrides,
        stage2_base_yaml=args.stage2_base_yaml,
        stage2_output_dir=args.stage2_output_dir,
        stage2_train_script=args.stage2_train_script,
        stage2_cfg_var_file=args.stage2_cfg_var_file,
        stage2_cfg_experiment_index=args.stage2_cfg_index,
        stage2_overrides=args.stage2_overrides,
        stage2_grid_base_yaml=args.stage2_grid_base_yaml,
    )


if __name__ == "__main__":
    main()