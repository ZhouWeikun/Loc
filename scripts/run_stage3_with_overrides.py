#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate temporary Stage-3 YAMLs, apply overrides, then run test or train.

Examples:
  python scripts/run_stage3_with_overrides.py \
    --base-yaml trainer_depends/configs/stage3_wingtra.yaml \
    --set selected_scene_name=zurich \
    --set load_stage2_ckpt=/abs/path/stage2.pth

  python scripts/run_stage3_with_overrides.py \
    --cfg-var-file scripts/stage3_cfg_wingtra.txt \
    --base-yaml trainer_depends/configs/stage3_wingtra.yaml

  python scripts/run_stage3_with_overrides.py \
    --cfg-var-file scripts/stage3_cfg_visloc.txt \
    --dry-run
"""

import argparse
import copy
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from scripts.run_stage1_with_overrides import (
    _build_generated_yaml_path,
    _build_subprocess_env,
    _dump_yaml,
    _format_override_pairs,
    _load_cfg_var_experiments,
    _load_yaml,
    _normalize_override_pairs,
    _set_nested,
)


DEFAULT_BASE_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "stage3_visloc.yaml")
DEFAULT_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_stage3_driver.py")
DEFAULT_TMP_DIR = os.path.join(PROJECT_ROOT, "gen_fm_exps", "run_yamls")
DEFAULT_CFG_VAR_FILE = os.path.join(PROJECT_ROOT, "scripts", "stage3_cfg_visloc.txt")


def _resolve_path(path):
    if not path:
        return ""
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, str(path)))


def _merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_merged_base_config(base_yaml, override_yamls):
    base_config = _load_yaml(base_yaml)
    for override_yaml in override_yamls:
        _merge_dict(base_config, _load_yaml(override_yaml))
    return base_config


def _run_one_experiment(
    base_config,
    output_dir,
    train_script,
    override_pairs,
    python_bin,
    dry_run,
    test_only,
    forward_args=None,
    save_yaml=None,
):
    run_config = copy.deepcopy(base_config)
    for dotted_key, value in override_pairs:
        _set_nested(run_config, dotted_key, value)

    generated_yaml = _resolve_path(save_yaml) if save_yaml else _build_generated_yaml_path(run_config, output_dir)
    _dump_yaml(generated_yaml, run_config)

    cmd = [python_bin, train_script]
    if test_only:
        cmd.append("--test_only")
    cmd.extend(["--p_yaml", generated_yaml])
    if forward_args:
        cmd.extend(forward_args)

    print(f"[Generated YAML] {generated_yaml}")
    print("[Overrides]")
    for item in _format_override_pairs(override_pairs):
        print(f"  - {item}")
    if forward_args:
        print("[Forward Args]")
        for item in forward_args:
            print(f"  - {item}")
    print(f"[Mode] {'test_only' if test_only else 'train'}")
    print(f"[Command] {' '.join(cmd)}")

    if dry_run:
        return {
            "generated_yaml": generated_yaml,
            "cmd": cmd,
            "override_pairs": list(override_pairs),
            "forward_args": list(forward_args or []),
            "test_only": bool(test_only),
            "dry_run": True,
        }

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_build_subprocess_env(python_bin), check=True)
    return {
        "generated_yaml": generated_yaml,
        "cmd": cmd,
        "override_pairs": list(override_pairs),
        "forward_args": list(forward_args or []),
        "test_only": bool(test_only),
        "dry_run": False,
    }


def run_stage3_with_overrides(
    base_yaml=DEFAULT_BASE_YAML,
    output_dir=DEFAULT_TMP_DIR,
    train_script=DEFAULT_TRAIN_SCRIPT,
    python_bin=sys.executable,
    overrides=None,
    cfg_var_file=None,
    dry_run=False,
    override_yamls=None,
    test_only=True,
    forward_args=None,
    save_yaml=None,
):
    """
    Callable entry for launching Stage-3 experiments.

    Args:
        base_yaml: Base Stage-3 YAML config path.
        output_dir: Directory used to store generated YAML files.
        train_script: Stage-3 driver script path.
        python_bin: Python executable used to launch Stage-3.
        overrides: One experiment's overrides.
        cfg_var_file: Batch experiment file path.
        dry_run: If True, only generate YAMLs and print commands.
        override_yamls: Extra YAML files merged after base_yaml.
        test_only: Whether to launch the Stage-3 entry with --test_only.
        forward_args: Extra CLI args appended after --p_yaml <generated_yaml>.
        save_yaml: Optional fixed output YAML path for single-run mode.

    Returns:
        A list of result dicts, one per launched experiment.
    """
    base_yaml = _resolve_path(base_yaml)
    train_script = _resolve_path(train_script)
    output_dir = _resolve_path(output_dir)
    python_bin = os.path.abspath(python_bin) if os.path.sep in str(python_bin) else python_bin
    override_yamls = [_resolve_path(path) for path in (override_yamls or [])]
    forward_args = list(forward_args or [])
    save_yaml = _resolve_path(save_yaml) if save_yaml else None

    if not os.path.exists(base_yaml):
        raise FileNotFoundError(f"Base YAML not found: {base_yaml}")
    for override_yaml in override_yamls:
        if not os.path.exists(override_yaml):
            raise FileNotFoundError(f"Override YAML not found: {override_yaml}")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if os.path.sep in str(python_bin) and not os.path.exists(python_bin):
        raise FileNotFoundError(f"Python executable not found: {python_bin}")

    base_config = _load_merged_base_config(base_yaml, override_yamls)
    print(f"[Base YAML] {base_yaml}")
    if override_yamls:
        print("[Override YAMLs]")
        for override_yaml in override_yamls:
            print(f"  - {override_yaml}")

    results = []
    if cfg_var_file:
        if save_yaml:
            raise ValueError("save_yaml is only supported in single-run mode, not with cfg_var_file.")
        cfg_var_file = _resolve_path(cfg_var_file)
        if not os.path.exists(cfg_var_file):
            raise FileNotFoundError(f"cfg_var file not found: {cfg_var_file}")
        experiments = _load_cfg_var_experiments(cfg_var_file, base_config)
        print(f"[cfg_var_file] {cfg_var_file}")
        print(f"[Experiments] {len(experiments)}")
        for idx, (lineno, override_pairs) in enumerate(experiments, start=1):
            print(f"\n{'=' * 80}")
            print(f"[Experiment {idx}/{len(experiments)}] source_line={lineno}")
            result = _run_one_experiment(
                base_config=base_config,
                output_dir=output_dir,
                train_script=train_script,
                override_pairs=override_pairs,
                python_bin=python_bin,
                dry_run=dry_run,
                test_only=test_only,
                forward_args=forward_args,
            )
            result["source_line"] = lineno
            result["experiment_idx"] = idx
            result["num_experiments"] = len(experiments)
            results.append(result)
        return results

    override_pairs = _normalize_override_pairs(overrides)
    if not override_pairs:
        print("[Overrides] none")
    results.append(
        _run_one_experiment(
            base_config=base_config,
            output_dir=output_dir,
            train_script=train_script,
            override_pairs=override_pairs,
            python_bin=python_bin,
            dry_run=dry_run,
            test_only=test_only,
            forward_args=forward_args,
            save_yaml=save_yaml,
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply YAML overrides for Stage-3 and run scripts/run_stage3_driver.py"
    )
    parser.add_argument(
        "--base-yaml",
        default=DEFAULT_BASE_YAML,
        help="Base Stage-3 YAML config path. Default: trainer_depends/configs/stage3_visloc.yaml",
    )
    parser.add_argument(
        "--override-yaml",
        dest="override_yamls",
        action="append",
        default=[],
        help="Extra YAML merged after base_yaml. Can be used multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_TMP_DIR,
        help="Directory used to store generated YAML files.",
    )
    parser.add_argument(
        "--save-yaml",
        default=None,
        help="Optional fixed output YAML path for single-run mode.",
    )
    parser.add_argument(
        "--train-script",
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to the Stage-3 driver script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the Stage-3 script.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in the form section.key=value. Can be used multiple times.",
    )
    parser.add_argument(
        "--cfg-var-file",
        default=None,
        help=(
            "Batch experiment file. Each non-empty line defines one experiment via comma-separated "
            f"key=value pairs. Suggested path: {DEFAULT_CFG_VAR_FILE}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate YAMLs and print the command, without launching Stage-3.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run Stage-3 train mode instead of the default test_only mode.",
    )
    parser.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Any args after '--' are forwarded to the Stage-3 Python entry.",
    )
    args = parser.parse_args()

    forward_args = list(args.forward_args)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    run_stage3_with_overrides(
        base_yaml=args.base_yaml,
        override_yamls=args.override_yamls,
        output_dir=args.output_dir,
        save_yaml=args.save_yaml,
        train_script=args.train_script,
        python_bin=args.python_bin,
        overrides=args.overrides,
        cfg_var_file=args.cfg_var_file,
        dry_run=args.dry_run,
        test_only=not args.train,
        forward_args=forward_args,
    )


if __name__ == "__main__":
    main()
