#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a temporary Stage-1 YAML from a base config, apply overrides, then run training.

Examples:
  python scripts/run_stage1_with_overrides.py \
    --set exp_setting.exp_name=stage1_dinov3_g2m \
    --set network_setting.backbone=dinov3 \
    --set network_setting.aggregator_type=g2m

  python scripts/run_stage1_with_overrides.py \
    --set exp_setting.exp_name=stage1_sat_query \
    --set data_setting.sat_as_query=true \
    --set learning_setting.num_epochs=20

  python scripts/run_stage1_with_overrides.py --dry-run

Python usage:
  from scripts.run_stage1_with_overrides import run_stage1_with_overrides

  run_stage1_with_overrides(
      overrides={
          "exp_setting.exp_name": "stage1_sat_query",
          "data_setting.sat_as_query": True,
      },
      cfg_var_file=None,
      dry_run=True,
  )
"""

import argparse
import copy
import os
import subprocess
import sys
from datetime import datetime

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "stage1_visual_encoder_visloc.yaml")
DEFAULT_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "trainers", "stage1_visual_encoder_wANCE.py")
DEFAULT_TMP_DIR = os.path.join(PROJECT_ROOT, "gen_fm_exps", "run_yamls")
DEFAULT_CFG_VAR_FILE = os.path.join(PROJECT_ROOT, "scripts", "stage1_cfg_visloc.txt")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path, config):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=False, sort_keys=False)


def _parse_override(override_text):
    if "=" not in override_text:
        raise ValueError(f"Invalid override: {override_text!r}. Expected format: section.key=value")
    key_path, raw_value = override_text.split("=", 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Invalid override: {override_text!r}. Empty key path.")
    value = yaml.safe_load(raw_value)
    return key_path, value


def _build_key_index(config):
    key_index = {}
    for section_name, section_value in config.items():
        if not isinstance(section_value, dict):
            continue
        for key in section_value.keys():
            key_index.setdefault(key, []).append(section_name)
    return key_index


def _resolve_leaf_key_to_dotted_key(key, key_index):
    sections = key_index.get(key, [])
    if len(sections) == 1:
        return f"{sections[0]}.{key}"
    if len(sections) == 0:
        raise KeyError(f"Cannot infer config section for key: {key!r}")
    raise KeyError(f"Ambiguous key {key!r}, found in sections: {sections}")


def _parse_cfg_var_line(line, key_index):
    overrides = []
    for item in line.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid cfg_var item: {item!r}. Expected key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        dotted_key = key if "." in key else _resolve_leaf_key_to_dotted_key(key, key_index)
        value = yaml.safe_load(raw_value)
        overrides.append((dotted_key, value))
    if not overrides:
        raise ValueError("Empty experiment config line.")
    return overrides


def _load_cfg_var_experiments(path, base_config):
    key_index = _build_key_index(base_config)
    experiments = []
    buffered_parts = []
    experiment_start_lineno = None

    def flush_buffer(end_lineno):
        nonlocal buffered_parts, experiment_start_lineno
        if not buffered_parts:
            experiment_start_lineno = None
            return
        merged_line = ",".join(buffered_parts).strip().rstrip(",").strip()
        if not merged_line:
            buffered_parts = []
            experiment_start_lineno = None
            return
        overrides = _parse_cfg_var_line(merged_line, key_index)
        experiments.append((experiment_start_lineno or end_lineno, overrides))
        buffered_parts = []
        experiment_start_lineno = None

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if experiment_start_lineno is None:
                experiment_start_lineno = lineno

            if ";" in line:
                segments = line.split(";")
                for idx, segment in enumerate(segments):
                    segment = segment.strip()
                    if segment:
                        buffered_parts.append(segment)
                    if idx < len(segments) - 1:
                        flush_buffer(lineno)
                        if idx < len(segments) - 2 or segments[-1].strip():
                            experiment_start_lineno = lineno
                continue

            buffered_parts.append(line)
            if experiment_start_lineno == lineno and not line.endswith(","):
                flush_buffer(lineno)

    flush_buffer(lineno if 'lineno' in locals() else 0)
    return experiments


def _set_nested(config, dotted_key, value):
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or cursor[key] is None:
            cursor[key] = {}
        if not isinstance(cursor[key], dict):
            raise TypeError(f"Cannot set nested key under non-dict node: {key!r} in {dotted_key!r}")
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _resolve_exp_name(config):
    exp_setting = config.get("exp_setting", {})
    exp_name = exp_setting.get("exp_name", None)
    if exp_name is None:
        return "stage1_run"
    return str(exp_name)


def _build_generated_yaml_path(config, output_dir):
    exp_name = _resolve_exp_name(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exp_name}_{timestamp}.yaml"
    return os.path.join(output_dir, filename)


def _format_override_pairs(override_pairs):
    return [f"{key}={value}" for key, value in override_pairs]


def _build_subprocess_env(python_bin):
    env = os.environ.copy()
    abs_python = os.path.abspath(python_bin) if os.path.sep in str(python_bin) else None
    if abs_python and abs_python.endswith("/bin/python"):
        env_root = os.path.dirname(os.path.dirname(abs_python))
        env_lib = os.path.join(env_root, "lib")
        if os.path.isdir(env_lib):
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{env_lib}:{old_ld}" if old_ld else env_lib
    return env


def _run_one_experiment(base_config, output_dir, train_script, override_pairs, python_bin, dry_run):
    run_config = copy.deepcopy(base_config)
    for dotted_key, value in override_pairs:
        _set_nested(run_config, dotted_key, value)

    generated_yaml = _build_generated_yaml_path(run_config, output_dir)
    _dump_yaml(generated_yaml, run_config)

    cmd = [python_bin, train_script, "--p_yaml", generated_yaml]

    print(f"[Generated YAML] {generated_yaml}")
    print("[Overrides]")
    for item in _format_override_pairs(override_pairs):
        print(f"  - {item}")
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


def _normalize_override_pairs(overrides):
    if overrides is None:
        return []
    if isinstance(overrides, dict):
        return list(overrides.items())

    normalized = []
    for item in overrides:
        if isinstance(item, str):
            normalized.append(_parse_override(item))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append((str(item[0]), item[1]))
            continue
        raise TypeError(
            "Unsupported overrides item. Expected 'section.key=value', "
            "(section.key, value), or a dict."
        )
    return normalized


def run_stage1_with_overrides(
    base_yaml=DEFAULT_BASE_YAML,
    output_dir=DEFAULT_TMP_DIR,
    train_script=DEFAULT_TRAIN_SCRIPT,
    python_bin=sys.executable,
    overrides=None,
    cfg_var_file=DEFAULT_CFG_VAR_FILE,
    dry_run=False,
):
    """
    Callable entry for launching Stage-1 experiments.

    Args:
        base_yaml: Base YAML config path.
        output_dir: Directory used to store generated YAML files.
        train_script: Stage-1 training entry script path.
        python_bin: Python executable used to launch training.
        overrides: One experiment's overrides. Supports:
            - dict[str, Any]
            - list[str] with "section.key=value"
            - list[tuple[str, Any]]
        cfg_var_file: Batch experiment file path. If truthy, batch mode is used.
        dry_run: If True, only generate YAMLs and print commands.

    Returns:
        A list of result dicts, one per launched experiment.
    """
    base_yaml = os.path.abspath(base_yaml)
    train_script = os.path.abspath(train_script)
    output_dir = os.path.abspath(output_dir)
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
        cfg_var_file = os.path.abspath(cfg_var_file)
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
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply YAML overrides for Stage-1 and run the selected Stage-1 training entry script"
    )
    parser.add_argument(
        "--base-yaml",
        default=DEFAULT_BASE_YAML,
        help="Base YAML config path. Default: trainer_depends/configs/stage1_visual_encoder_visloc.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_TMP_DIR,
        help="Directory used to store generated YAML files.",
    )
    parser.add_argument(
        "--train-script",
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to the Stage-1 training entry script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the Stage-1 training script.",
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
        default=DEFAULT_CFG_VAR_FILE,
        help="Batch experiment file. Each non-empty line defines one experiment via comma-separated key=value pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate the YAML and print the command, without launching training.",
    )
    args = parser.parse_args()
    run_stage1_with_overrides(
        base_yaml=args.base_yaml,
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

# CLI example:
#   python scripts/run_stage1_with_overrides.py \
#     --set exp_setting.exp_name=stage1_sat_query \
#     --set data_setting.sat_as_query=true \
#     --set learning_setting.num_epochs=20
#
# CLI batch example:
#   python scripts/run_stage1_with_overrides.py  --cfg-var-file gen_fm_exps/run_yamls/stage1_cfg_var.txt
# CLI batch excample for ctrl exp:
#     python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage1_with_overrides.py  \
#     --train-script /home/data/zwk/pyproj_neuloc_v0/trainers/stage1_visual_encoder_controlexps.py \
#     --base-yaml /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder_visloc_control_exps.yaml\
#     --cfg-var-file /home/data/zwk/pyproj_neuloc_v0/scripts/stage1_cfg_visloc_control_exps.txt \


# Python-call example:
#   from scripts.run_stage1_with_overrides import run_stage1_with_overrides
#   run_stage1_with_overrides(
#       overrides={
#           "exp_setting.exp_name": "stage1_sat_query",
#           "data_setting.sat_as_query": True,
#           "learning_setting.num_epochs": 20,
#       },
#       cfg_var_file=None,
#       dry_run=True,
#   )
#
# Python-call batch example:
#   from scripts.run_stage1_with_overrides import run_stage1_with_overrides
#   run_stage1_with_overrides(
#       cfg_var_file="gen_fm_exps/run_yamls/stage1_cfg_var.txt",
#       dry_run=False,
#   )
