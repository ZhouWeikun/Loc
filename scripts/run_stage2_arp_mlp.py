#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a temporary Stage-2 APR MLP YAML, apply overrides, then run training.

Examples:
  python scripts/run_stage2_arp_mlp.py \
    --set exp_setting.load_stage1_ckpt=/path/to/stage1_epoch.pth \
    --set exp_setting.inherit_stage1_yaml=/path/to/stage1_opts.yaml \
    --set exp_setting.exp_name=stage2_apr_mlp_visloc03

  python scripts/run_stage2_arp_mlp.py --dry-run
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


DEFAULT_BASE_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "stage2_arp_mlp.yaml")
DEFAULT_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "trainers", "stage2_arp_mlp.py")
DEFAULT_TMP_DIR = os.path.join(PROJECT_ROOT, "gen_fm_exps", "run_yamls")
DEFAULT_CFG_VAR_FILE = os.path.join(PROJECT_ROOT, "scripts", "stage2_arp_cfg_var.txt")


def _resolve_path(path):
    if not path:
        return ""
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, str(path)))


def _run_one_experiment(base_config, output_dir, train_script, override_pairs, python_bin, dry_run, test_only=False):
    run_config = copy.deepcopy(base_config)
    for dotted_key, value in override_pairs:
        _set_nested(run_config, dotted_key, value)

    generated_yaml = _build_generated_yaml_path(run_config, output_dir)
    _dump_yaml(generated_yaml, run_config)

    cmd = [python_bin, train_script]
    if test_only:
        cmd.append("--test_only")
    cmd.extend(["--p_yaml", generated_yaml])

    print(f"[Generated YAML] {generated_yaml}")
    print("[Overrides]")
    if override_pairs:
        for item in _format_override_pairs(override_pairs):
            print(f"  - {item}")
    else:
        print("  none")
    print(f"[Command] {' '.join(cmd)}")

    if dry_run:
        return {
            "generated_yaml": generated_yaml,
            "cmd": cmd,
            "override_pairs": list(override_pairs),
            "dry_run": True,
        }

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_build_subprocess_env(python_bin), check=True)
    return {
        "generated_yaml": generated_yaml,
        "cmd": cmd,
        "override_pairs": list(override_pairs),
        "dry_run": False,
    }


def run_stage2_arp_mlp(
        base_yaml=DEFAULT_BASE_YAML,
        output_dir=DEFAULT_TMP_DIR,
        train_script=DEFAULT_TRAIN_SCRIPT,
        python_bin=sys.executable,
        overrides=None,
        cfg_var_file=None,
        dry_run=False,
        test_only=False,
):
    base_yaml = _resolve_path(base_yaml)
    train_script = _resolve_path(train_script)
    output_dir = _resolve_path(output_dir)
    python_bin = os.path.abspath(python_bin) if os.path.sep in str(python_bin) else python_bin

    if not os.path.exists(base_yaml):
        raise FileNotFoundError(f"Base YAML not found: {base_yaml}")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if os.path.sep in str(python_bin) and not os.path.exists(python_bin):
        raise FileNotFoundError(f"Python executable not found: {python_bin}")

    base_config = _load_yaml(base_yaml)
    print(f"[Base YAML] {base_yaml}")

    results = []
    if cfg_var_file:
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
            )
            result["source_line"] = lineno
            result["experiment_idx"] = idx
            result["num_experiments"] = len(experiments)
            results.append(result)
        return results

    override_pairs = _normalize_override_pairs(overrides)
    results.append(
        _run_one_experiment(
            base_config=base_config,
            output_dir=output_dir,
            train_script=train_script,
            override_pairs=override_pairs,
            python_bin=python_bin,
            dry_run=dry_run,
            test_only=test_only,
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Apply YAML overrides and run trainers/stage2_arp_mlp.py")
    parser.add_argument("--base-yaml", default=DEFAULT_BASE_YAML)
    parser.add_argument("--output-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in the form section.key=value. Can be used multiple times.",
    )
    parser.add_argument("--cfg-var-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    run_stage2_arp_mlp(
        base_yaml=args.base_yaml,
        output_dir=args.output_dir,
        train_script=args.train_script,
        python_bin=args.python_bin,
        overrides=args.overrides,
        cfg_var_file=args.cfg_var_file,
        dry_run=args.dry_run,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
  # /root/miniconda3/envs/neuloc_wisp/bin/python scripts/run_stage2_arp_mlp.py \
  #   --set scenes_setting.selected_scene_name=zurich \
  #   --set exp_setting.load_stage1_ckpt=/path/to/stage1_epoch.pth \
  #   --set exp_setting.inherit_stage1_yaml=/path/to/stage1_opts.yaml \
  #   --set exp_setting.exp_name=stage2_apr_mlp_zurich