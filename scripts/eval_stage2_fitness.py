#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Stage-2 retrieval/localization fitness with aligned Stage-1-style settings.

Examples:
  /root/miniconda3/envs/neuloc_wisp/bin/python scripts/util_eval_stage2_fitness.py \
    --p_yaml gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_woSatQuery/opts.yaml \
    --stage2-ckpt gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_woSatQuery/epoch480.pth \
    --mode rc

  /root/miniconda3/envs/neuloc_wisp/bin/python scripts/util_eval_stage2_fitness.py \
    --p_yaml gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_woSatQuery/opts.yaml \
    --mode rc_rot_scale \
    --delta-rot-deg 10 \
    --n-scales 4 \
    --save-json gen_fm_exps/analysis/stage2_eval_epoch480.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict


def _maybe_reexec_with_env_lib():
    """Ensure the conda env lib dir is first in LD_LIBRARY_PATH before heavy imports."""
    if os.environ.get("_EVAL_STAGE2_FITNESS_BOOTSTRAPPED") == "1":
        return

    python_bin = os.path.abspath(sys.executable)
    if not python_bin.endswith("/bin/python"):
        return

    env_root = os.path.dirname(os.path.dirname(python_bin))
    env_lib = os.path.join(env_root, "lib")
    if not os.path.isdir(env_lib):
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    if ld_parts and ld_parts[0] == env_lib:
        return

    env = os.environ.copy()
    env["_EVAL_STAGE2_FITNESS_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else env_lib
    os.execve(python_bin, [python_bin] + sys.argv, env)


_maybe_reexec_with_env_lib()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DEFAULT_K_VALUES = (1, 5, 10, 20, 50, 256, 512, 1024)


def _parse_k_values(text):
    items = [item.strip() for item in str(text).split(",")]
    values = []
    for item in items:
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("k_values cannot be empty.")
    return tuple(values)


def _build_script_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-2 retrieval fitness with aligned eval settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--p_yaml",
        required=True,
        help="Stage-2 base YAML or experiment opts.yaml.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        default="",
        help="Stage-2 checkpoint path. If omitted, try to resolve automatically.",
    )
    parser.add_argument(
        "--stage1-ckpt",
        default="",
        help="Optional Stage-1 checkpoint override for vis_encoder/vis_aggregator.",
    )
    parser.add_argument(
        "--mode",
        default="rc",
        choices=("rc", "rc_rot", "rc_scale", "rc_rot_scale"),
        help="Gallery layout / evaluation mode.",
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Gallery overlap ratio.")
    parser.add_argument(
        "--fixed-scale",
        type=float,
        default=None,
        help="Fixed gallery scale for rc / rc_rot modes. Default uses dataset mean scale.",
    )
    parser.add_argument("--fixed-rot", type=float, default=0.0, help="Fixed gallery rotation for rc / rc_scale.")
    parser.add_argument("--delta-rot-deg", type=float, default=10.0, help="Rotation step for rot-aware modes.")
    parser.add_argument("--n-scales", type=int, default=4, help="Number of gallery scales for scale-aware modes.")
    parser.add_argument("--scale-mode", default="linear", choices=("linear", "log"), help="Gallery scale sampling mode.")
    parser.add_argument(
        "--use-train-uav",
        action="store_true",
        help="Evaluate on the training UAV split instead of the test split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Query batch size. Defaults to trainer batchsize_uav.",
    )
    parser.add_argument(
        "--num-workers-eval",
        type=int,
        default=None,
        help="Evaluation DataLoader workers. Defaults to opt.num_worker_eval or 0.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit the number of queries. Default uses the full split.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_K_VALUES),
        help="Comma-separated recall@k values.",
    )
    parser.add_argument(
        "--cache-gallery",
        action="store_true",
        help="Cache gallery bank to disk and reuse it when available.",
    )
    parser.add_argument(
        "--gallery-root-dir",
        default=None,
        help="Optional root dir for cached gallery banks.",
    )
    parser.add_argument(
        "--gallery-name-prefix",
        default=None,
        help="Optional gallery cache prefix passed to the trainer.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to write a compact JSON summary.",
    )
    return parser


def _parse_script_args(argv=None):
    parser = _build_script_arg_parser()
    return parser.parse_known_args(argv)


def _find_latest_epoch_ckpt(directory):
    if not directory or not os.path.isdir(directory):
        return None

    best = None
    for name in os.listdir(directory):
        match = re.fullmatch(r"epoch(\d+)\.pth", name)
        if not match:
            continue
        epoch = int(match.group(1))
        path = os.path.join(directory, name)
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return None if best is None else best[1]


def _resolve_cli_path(path_text):
    if not path_text:
        return ""
    if os.path.isabs(path_text):
        return path_text
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_text))


def _resolve_stage2_ckpt(script_args, opt):
    if script_args.stage2_ckpt:
        ckpt_path = _resolve_cli_path(script_args.stage2_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")
        return ckpt_path

    p_yaml_abs = _resolve_cli_path(script_args.p_yaml)
    if os.path.basename(p_yaml_abs) == "opts.yaml":
        ckpt_path = _find_latest_epoch_ckpt(os.path.dirname(p_yaml_abs))
        if ckpt_path is not None:
            return ckpt_path

    ckpt_root = getattr(opt, "dir2save_ckpt", None)
    exp_name = getattr(opt, "exp_name", None)
    if ckpt_root and exp_name:
        ckpt_dir = _resolve_cli_path(os.path.join(ckpt_root, exp_name))
        ckpt_path = _find_latest_epoch_ckpt(ckpt_dir)
        if ckpt_path is not None:
            return ckpt_path

    raise FileNotFoundError(
        "Unable to resolve a Stage-2 checkpoint. Pass --stage2-ckpt explicitly or use an opts.yaml directory "
        "that contains epoch*.pth files."
    )


def _resolve_mode_defaults(mode):
    defaults = {
        "rc": {
            "query_rot2uniform": True,
            "query_scale2uniform": False,
            "report_rot_error": False,
            "report_scale_error": False,
            "report_title": "Stage2 RC Localization",
        },
        "rc_rot": {
            "query_rot2uniform": False,
            "query_scale2uniform": False,
            "report_rot_error": True,
            "report_scale_error": False,
            "report_title": "Stage2 RC + Rot Localization",
        },
        "rc_scale": {
            "query_rot2uniform": True,
            "query_scale2uniform": False,
            "report_rot_error": False,
            "report_scale_error": True,
            "report_title": "Stage2 RC + Scale Localization",
        },
        "rc_rot_scale": {
            "query_rot2uniform": False,
            "query_scale2uniform": False,
            "report_rot_error": True,
            "report_scale_error": True,
            "report_title": "Stage2 RC + Rot + Scale Localization",
        },
    }
    return defaults[mode]


def _build_layout_cfg(script_args):
    from trainers.util_stage2_gallery_manager import Stage2ReferenceGalleryLayoutConfig

    kwargs = {
        "mode": script_args.mode,
        "overlap": script_args.overlap,
        "fixed_rot": script_args.fixed_rot,
        "fixed_scale": script_args.fixed_scale,
        "delta_rot_deg": script_args.delta_rot_deg,
        "n_scales": script_args.n_scales,
        "scale_mode": script_args.scale_mode,
    }
    return Stage2ReferenceGalleryLayoutConfig(**kwargs)


def _build_eval_cfg(trainer, script_args):
    mode_defaults = _resolve_mode_defaults(script_args.mode)
    batch_size = script_args.batch_size
    if batch_size is None:
        batch_size = int(getattr(trainer.opt, "batchsize_uav", 32))

    num_workers = script_args.num_workers_eval
    if num_workers is None:
        num_workers = int(getattr(trainer.opt, "num_worker_eval", 0))

    return trainer._make_stage2_retrieval_eval_cfg(
        use_train_uav=bool(script_args.use_train_uav),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        query_rot2uniform=bool(mode_defaults["query_rot2uniform"]),
        query_scale2uniform=bool(mode_defaults["query_scale2uniform"]),
        k_values=_parse_k_values(script_args.k_values),
        max_queries=script_args.max_queries,
        print_results=True,
        report_title=mode_defaults["report_title"],
        report_rot_error=bool(mode_defaults["report_rot_error"]),
        report_scale_error=bool(mode_defaults["report_scale_error"]),
    )


def _build_cli_summary(eval_res, stage2_ckpt, layout_cfg, eval_cfg):
    summary = {
        "scene_name": eval_res["scene_name"],
        "stage2_ckpt": stage2_ckpt,
        "n_queries": int(eval_res["n_queries"]),
        "layout_cfg": asdict(layout_cfg),
        "eval_cfg": {
            "use_train_uav": bool(eval_cfg.use_train_uav),
            "batch_size": int(eval_cfg.batch_size),
            "num_workers": int(eval_cfg.num_workers),
            "query_rot2uniform": bool(eval_cfg.query_rot2uniform),
            "query_scale2uniform": bool(eval_cfg.query_scale2uniform),
            "k_values": [int(k) for k in eval_cfg.k_values],
            "max_queries": None if eval_cfg.max_queries is None else int(eval_cfg.max_queries),
        },
        "recall@k": {str(int(k)): float(v) for k, v in eval_res["recall@k"].items()},
        "error_rc_norm": float(eval_res["error_rc_norm"]),
        "error_rc_meter": float(eval_res["error_rc_meter"]),
        "runtime_gallery_summary": eval_res["runtime_gallery_summary"],
    }
    if bool(eval_cfg.report_rot_error) and "error_rot_deg" in eval_res:
        summary["error_rot_deg"] = float(eval_res["error_rot_deg"])
    if bool(eval_cfg.report_scale_error) and "error_scale_normed" in eval_res:
        summary["error_scale_normed"] = float(eval_res["error_scale_normed"])
    return summary


def _print_final_summary(summary):
    recall_items = []
    for key in summary["eval_cfg"]["k_values"]:
        value = summary["recall@k"].get(str(int(key)))
        if value is None:
            continue
        recall_items.append(f"R@{int(key)}={float(value) * 100.0:.3f}%")

    print("")
    print("[Stage2 Fitness] scene={scene_name} | ckpt={ckpt} | N={n_queries}".format(
        scene_name=summary["scene_name"],
        ckpt=os.path.basename(summary["stage2_ckpt"]),
        n_queries=summary["n_queries"],
    ))
    print("[Stage2 Fitness] " + " | ".join(recall_items))
    print(
        "[Stage2 Fitness] error_rc_norm={error_rc_norm:.6f} | error_rc_meter={error_rc_meter:.3f}m".format(
            error_rc_norm=summary["error_rc_norm"],
            error_rc_meter=summary["error_rc_meter"],
        )
    )
    if "error_rot_deg" in summary:
        print("[Stage2 Fitness] error_rot_deg={:.3f}".format(summary["error_rot_deg"]))
    if "error_scale_normed" in summary:
        print("[Stage2 Fitness] error_scale_normed={:.6f}".format(summary["error_scale_normed"]))


def _write_json(path, payload):
    output_path = _resolve_cli_path(path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[Stage2 Fitness] saved json: {output_path}")


def main(argv=None):
    script_args, remaining_argv = _parse_script_args(argv)

    from trainer_depends.config.parser import get_parse
    from trainers.stage2_INGP import GridHashFitTrainer

    parser_argv = [sys.argv[0], "--p_yaml", script_args.p_yaml] + list(remaining_argv)
    sys.argv = parser_argv
    opt = get_parse()

    if script_args.stage1_ckpt:
        opt.load_stage1_ckpt = _resolve_cli_path(script_args.stage1_ckpt)
        if not os.path.exists(opt.load_stage1_ckpt):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {opt.load_stage1_ckpt}")
    opt.load2test = _resolve_stage2_ckpt(script_args, opt)

    stage1_ckpt_for_test = getattr(opt, "load_stage1_ckpt", "")
    opt.load_stage1_ckpt = ""
    trainer = GridHashFitTrainer(opt=opt)
    trainer.opt.load_stage1_ckpt = stage1_ckpt_for_test
    trainer._load_checkpoints_for_test()

    layout_cfg = _build_layout_cfg(script_args)
    eval_cfg = _build_eval_cfg(trainer, script_args)
    feature_cfg = trainer._make_stage2_gallery_feature_cfg()
    feature_cfg.build_faiss = True
    feature_cfg.show_progress = True

    gallery_state = trainer.eval_gallery_bank(
        layout_cfg=layout_cfg,
        feature_cfg=feature_cfg,
        retrieval_eval_cfg=eval_cfg,
        gallery_save_dir=None,
        load_if_exists=bool(script_args.cache_gallery),
        save_gallery=bool(script_args.cache_gallery),
        init_datasets=True,
        load_ckpt=False,
        gallery_root_dir=_resolve_cli_path(script_args.gallery_root_dir) if script_args.gallery_root_dir else None,
        gallery_name_prefix=script_args.gallery_name_prefix,
    )
    eval_res = gallery_state["eval_res"]

    summary = _build_cli_summary(
        eval_res=eval_res,
        stage2_ckpt=opt.load2test,
        layout_cfg=layout_cfg,
        eval_cfg=eval_cfg,
    )
    _print_final_summary(summary)

    if script_args.save_json:
        _write_json(script_args.save_json, summary)


if __name__ == "__main__":
    main()
    """命令行调用示例
    /root/miniconda3/envs/neuloc_wisp/bin/python scripts/eval_stage2_fitness.py \
    --p_yaml gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_woSatQuery/opts.yaml \
    --stage2-ckpt gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_woSatQuery/epoch480.pth \
    --mode rc \
    --save-json gen_fm_exps/analysis/stage2_eval_epoch480.json
    """
