#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stable Stage-3 launcher that bypasses the trainer file's __main__ block.

Modes:
1. Direct mode:
   python scripts/run_stage3_driver.py --test_only --p_yaml /abs/path/run.yaml
2. Batch mode:
   python scripts/run_stage3_driver.py --base-yaml ... --cfg-var-file ...

Direct mode reconstructs the runtime as:
  get_parse(...) -> MetricNetTrainer(opt=opt) -> test/train
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_STAGE3_YAML = os.path.join(PROJECT_ROOT, "trainer_depends", "configs", "stage3_visloc.yaml")
DEFAULT_DRIVER_PATH = os.path.join(PROJECT_ROOT, "scripts", "run_stage3_driver.py")


def _bootstrap_runtime_env():
    abs_python = os.path.abspath(sys.executable)
    if not abs_python.endswith("/bin/python"):
        return

    env_root = os.path.dirname(os.path.dirname(abs_python))
    env_lib = os.path.join(env_root, "lib")
    if not os.path.isdir(env_lib):
        return

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in old_ld.split(":") if part]
    if ld_parts[:1] == [env_lib]:
        return

    os.environ["LD_LIBRARY_PATH"] = f"{env_lib}:{old_ld}" if old_ld else env_lib
    if os.environ.get("_STAGE3_DRIVER_REEXECED") == "1":
        return

    os.environ["_STAGE3_DRIVER_REEXECED"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


def _should_delegate_to_override_runner(argv):
    batch_flags = {
        "--base-yaml",
        "--override-yaml",
        "--output-dir",
        "--save-yaml",
        "--python-bin",
        "--set",
        "--cfg-var-file",
        "--dry-run",
    }
    for arg in argv:
        if arg in batch_flags:
            return True
        if any(arg.startswith(flag + "=") for flag in batch_flags):
            return True
    return False


def _run_override_mode(argv):
    from scripts.run_stage3_with_overrides import run_stage3_with_overrides

    parser = argparse.ArgumentParser(
        description="Stage-3 batch override launcher"
    )
    parser.add_argument("--base-yaml", default=DEFAULT_STAGE3_YAML)
    parser.add_argument("--override-yaml", dest="override_yamls", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-yaml", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--cfg-var-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test_only", dest="test_only", action="store_true")
    parser.add_argument("--train", dest="test_only", action="store_false")
    parser.set_defaults(test_only=True)
    parser.add_argument("forward_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    forward_args = list(args.forward_args)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    kwargs = {
        "base_yaml": args.base_yaml,
        "train_script": DEFAULT_DRIVER_PATH,
        "python_bin": args.python_bin,
        "overrides": args.overrides,
        "cfg_var_file": args.cfg_var_file,
        "dry_run": args.dry_run,
        "override_yamls": args.override_yamls,
        "test_only": args.test_only,
        "forward_args": forward_args,
        "save_yaml": args.save_yaml,
    }
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir

    run_stage3_with_overrides(**kwargs)


def _run_direct_mode(argv):
    from trainer_depends.config.parser import get_parse
    from trainers.stage3_proxy_linearProjector_wANCE_evotorch import MetricNetTrainer

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", dest="test_only", action="store_true", help="Run Stage-3 test().")
    parser.add_argument("--train", dest="test_only", action="store_false", help="Run Stage-3 train().")
    parser.set_defaults(test_only=True)
    args, remaining_argv = parser.parse_known_args(argv)

    if "--p_yaml" not in remaining_argv:
        remaining_argv.extend(["--p_yaml", DEFAULT_STAGE3_YAML])

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining_argv
        opt = get_parse(print_summary=False)
        scenes_setting = getattr(opt, "scenes_setting", {}) or {}
        selected_scene_name = scenes_setting.get("selected_scene_name", None)
        scenes = scenes_setting.get("scenes", []) if isinstance(scenes_setting, dict) else []
        scene_names = [scene.get("name", "<unnamed>") for scene in scenes if isinstance(scene, dict)]
        print(
            f"[Stage3 Driver] selected_scene_name={selected_scene_name}, "
            f"scenes={scene_names}"
        )
        if selected_scene_name and scene_names:
            if len(scene_names) != 1 or scene_names[0] != selected_scene_name:
                raise ValueError(
                    "Stage3 scene selection mismatch: "
                    f"selected_scene_name={selected_scene_name}, expanded_scenes={scene_names}"
                )
        trainer = MetricNetTrainer(opt=opt)
        if args.test_only:
            trainer.test(use_train_uav=False)
        else:
            trainer.train()
    finally:
        sys.argv = original_argv


def main():
    _bootstrap_runtime_env()
    argv = sys.argv[1:]
    if _should_delegate_to_override_runner(argv):
        _run_override_mode(argv)
        return
    _run_direct_mode(argv)


if __name__ == "__main__":
    main()
  # python /home/data/zwk/pyproj_neuloc_v0/scripts/run_stage3_driver.py \
  #   --base-yaml  /home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_wingtra.yaml \
  #   --cfg-var-file  /home/data/zwk/pyproj_neuloc_v0/scripts/stage3_cfg_wingtra.txt
