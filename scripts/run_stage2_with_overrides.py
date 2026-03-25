#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate temporary Stage-2 YAMLs, optionally patch a temporary grid YAML, then run training.

Examples:
  python scripts/run_stage2_with_overrides.py \
    --set exp_setting.exp_name=stage2_visloc_debug \
    --set learning_setting.num_epochs=50 \
    --set hardware_setting.autocast=true

  python scripts/run_stage2_with_overrides.py \
    --set exp_setting.exp_name=stage2_visloc_codebook20 \
    --set gridcfg.codebook_bitwidth=20 \
    --set gridcfg.max_grid_res=768 \
    --dry-run

  python scripts/run_stage2_with_overrides.py --cfg-var-file gen_fm_exps/run_yamls/stage2_cfg_var.txt

Python usage:
  from scripts.run_stage2_with_overrides import run_stage2_with_overrides

  run_stage2_with_overrides(
      overrides={
          "exp_setting.exp_name": "stage2_visloc_debug",
          "learning_setting.num_epochs": 50,
          "gridcfg.codebook_bitwidth": 20,
      },
      dry_run=True,
  )
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


DEFAULT_BASE_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "stage2_INGP_visloc.yaml")
DEFAULT_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "trainers", "stage2_INGP.py")
DEFAULT_TMP_DIR = os.path.join(PROJECT_ROOT, "gen_fm_exps", "run_yamls")
DEFAULT_CFG_VAR_FILE = os.path.join(PROJECT_ROOT, "gen_fm_exps", "run_yamls", "stage2_cfg_var.txt")
DEFAULT_GRID_BASE_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "nerf_hash_visloc.yaml")
GRIDCFG_PREFIX = "gridcfg."


def _resolve_path(path):
    if not path:
        return ""
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, str(path)))


def _split_override_pairs(override_pairs):
    stage2_pairs = []
    grid_pairs = []
    for key, value in override_pairs:
        key = str(key)
        if key.startswith(GRIDCFG_PREFIX):
            grid_key = key[len(GRIDCFG_PREFIX):].strip()
            if not grid_key:
                raise ValueError(f"Invalid grid override key: {key!r}")
            # Shorthand: gridcfg.codebook_bitwidth -> grid.codebook_bitwidth
            if "." not in grid_key:
                grid_key = f"grid.{grid_key}"
            grid_pairs.append((grid_key, value))
        else:
            stage2_pairs.append((key, value))
    return stage2_pairs, grid_pairs


def _resolve_grid_base_yaml(run_config, explicit_grid_base_yaml=None):
    if explicit_grid_base_yaml:
        candidate = explicit_grid_base_yaml
    else:
        candidate = (run_config.get("network_setting") or {}).get("p_grid_config_yaml", None)
    if not candidate:
        candidate = DEFAULT_GRID_BASE_YAML
    grid_base_yaml = _resolve_path(candidate)
    if not os.path.exists(grid_base_yaml):
        raise FileNotFoundError(f"Grid base YAML not found: {grid_base_yaml}")
    return grid_base_yaml


def _build_generated_grid_yaml_path(stage2_yaml_path):
    stem, ext = os.path.splitext(stage2_yaml_path)
    ext = ext or ".yaml"
    return f"{stem}__grid{ext}"


def _run_one_experiment(
    base_config,
    output_dir,
    train_script,
    override_pairs,
    python_bin,
    dry_run,
    grid_base_yaml=None,
):
    run_config = copy.deepcopy(base_config)
    if grid_base_yaml:
        _set_nested(run_config, "network_setting.p_grid_config_yaml", _resolve_path(grid_base_yaml))

    stage2_pairs, grid_pairs = _split_override_pairs(override_pairs)
    for dotted_key, value in stage2_pairs:
        _set_nested(run_config, dotted_key, value)

    generated_yaml = _build_generated_yaml_path(run_config, output_dir)
    generated_grid_yaml = None
    resolved_grid_base_yaml = _resolve_grid_base_yaml(run_config)

    if grid_pairs:
        grid_config = _load_yaml(resolved_grid_base_yaml)
        for dotted_key, value in grid_pairs:
            _set_nested(grid_config, dotted_key, value)
        generated_grid_yaml = _build_generated_grid_yaml_path(generated_yaml)
        _dump_yaml(generated_grid_yaml, grid_config)
        _set_nested(run_config, "network_setting.p_grid_config_yaml", generated_grid_yaml)

    _dump_yaml(generated_yaml, run_config)

    cmd = [python_bin, train_script, "--p_yaml", generated_yaml]

    print(f"[Generated YAML] {generated_yaml}")
    if generated_grid_yaml is not None:
        print(f"[Generated Grid YAML] {generated_grid_yaml}")
        print(f"[Grid Base YAML] {resolved_grid_base_yaml}")
    print("[Overrides]")
    for item in _format_override_pairs(stage2_pairs):
        print(f"  - {item}")
    if grid_pairs:
        print("[Grid Overrides]")
        for item in _format_override_pairs(grid_pairs):
            print(f"  - {item}")
    print(f"[Command] {' '.join(cmd)}")

    if dry_run:
        return {
            "generated_yaml": generated_yaml,
            "generated_grid_yaml": generated_grid_yaml,
            "grid_base_yaml": resolved_grid_base_yaml,
            "cmd": cmd,
            "override_pairs": list(stage2_pairs),
            "grid_override_pairs": list(grid_pairs),
            "dry_run": True,
        }

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_build_subprocess_env(python_bin), check=True)
    return {
        "generated_yaml": generated_yaml,
        "generated_grid_yaml": generated_grid_yaml,
        "grid_base_yaml": resolved_grid_base_yaml,
        "cmd": cmd,
        "override_pairs": list(stage2_pairs),
        "grid_override_pairs": list(grid_pairs),
        "dry_run": False,
    }


def run_stage2_with_overrides(
    base_yaml=DEFAULT_BASE_YAML,
    output_dir=DEFAULT_TMP_DIR,
    train_script=DEFAULT_TRAIN_SCRIPT,
    python_bin=sys.executable,
    overrides=None,
    cfg_var_file=None,
    dry_run=False,
    grid_base_yaml=None,
):
    """
    Callable entry for launching Stage-2 experiments.

    Args:
        base_yaml: Base Stage-2 YAML config path.
        output_dir: Directory used to store generated YAML files.
        train_script: Stage-2 training entry script path.
        python_bin: Python executable used to launch training.
        overrides: One experiment's overrides. Supports:
            - dict[str, Any]
            - list[str] with "section.key=value"
            - list[tuple[str, Any]]
          Grid-YAML overrides must use the prefix 'gridcfg.'.
          Example: 'gridcfg.codebook_bitwidth=20'.
        cfg_var_file: Batch experiment file path. When omitted, run a single experiment.
        dry_run: If True, only generate YAMLs and print commands.
        grid_base_yaml: Optional nerf_hash YAML used as the base for temporary grid configs.

    Returns:
        A list of result dicts, one per launched experiment.
    """
    base_yaml = _resolve_path(base_yaml)
    train_script = _resolve_path(train_script)
    output_dir = _resolve_path(output_dir)
    python_bin = os.path.abspath(python_bin) if os.path.sep in str(python_bin) else python_bin
    grid_base_yaml = _resolve_path(grid_base_yaml) if grid_base_yaml else None

    if not os.path.exists(base_yaml):
        raise FileNotFoundError(f"Base YAML not found: {base_yaml}")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if os.path.sep in str(python_bin) and not os.path.exists(python_bin):
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    if grid_base_yaml and not os.path.exists(grid_base_yaml):
        raise FileNotFoundError(f"Grid base YAML not found: {grid_base_yaml}")

    base_config = _load_yaml(base_yaml)
    print(f"[Base YAML] {base_yaml}")
    if grid_base_yaml:
        print(f"[Grid Base YAML Override] {grid_base_yaml}")

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
                grid_base_yaml=grid_base_yaml,
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
            grid_base_yaml=grid_base_yaml,
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply YAML overrides for Stage-2, optionally patch a temporary grid YAML, and run trainers/stage2_INGP.py"
    )
    parser.add_argument(
        "--base-yaml",
        default=DEFAULT_BASE_YAML,
        help="Base Stage-2 YAML config path. Default: trainer_depends/configs/stage2_INGP_visloc.yaml",
    )
    parser.add_argument(
        "--grid-base-yaml",
        default=None,
        help=(
            "Optional base nerf_hash YAML used when creating temporary grid configs. "
            "Defaults to network_setting.p_grid_config_yaml from the Stage-2 YAML."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_TMP_DIR,
        help="Directory used to store generated YAML files.",
    )
    parser.add_argument(
        "--train-script",
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to the Stage-2 training entry script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the Stage-2 training script.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Override in the form section.key=value. Can be used multiple times. "
            "Use gridcfg.<key>=value to patch the temporary grid YAML; "
            "for example gridcfg.codebook_bitwidth=20 or gridcfg.blas.level=4."
        ),
    )
    parser.add_argument(
        "--cfg-var-file",
        default=None,
        help=(
            "Batch experiment file. Each non-empty line defines one experiment via comma-separated "
            "key=value pairs. Leave unset for single-run mode. Suggested path: "
            f"{DEFAULT_CFG_VAR_FILE}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate YAMLs and print the command, without launching training.",
    )
    args = parser.parse_args()

    run_stage2_with_overrides(
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
    #todo:记得检查最顶上的默认配置
    main()
    # Function-call example with cfg_var_file,用完后记得注销：
    # from scripts.run_stage2_with_overrides import run_stage2_with_overrides
    # run_stage2_with_overrides(
    #     cfg_var_file="gen_fm_exps/run_yamls/stage2_cfg_var.txt",
    #     dry_run=False,
    # )
    # python run_stage2_with_overrides.py  --cfg-var-file tmp_run_yamls/stage2_cfg_var.txt
