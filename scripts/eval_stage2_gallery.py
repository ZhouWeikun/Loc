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


def _parse_n_bins_4d(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    norm_text = text.lower().replace("x", ",")
    items = [item.strip() for item in norm_text.split(",") if item.strip()]
    if len(items) != 4:
        raise ValueError(f"n_bins_4d must provide 4 integers, got: {text!r}")
    values = tuple(int(item) for item in items)
    if any(v <= 0 for v in values):
        raise ValueError(f"n_bins_4d entries must be > 0, got: {values}")
    return values


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
        choices=("rc", "rc_rot", "rc_scale", "rc_rot_scale", "n_bins_4d"),
        help="Gallery layout / evaluation mode.",
    )
    parser.add_argument(
        "--n-bins-4d",
        type=str,
        default="",
        help="Direct gallery bins as nr,nc,rot,scale or nrxncxrotxscale. Only used when --mode n_bins_4d.",
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
        "--search-backend",
        default="faiss",
        choices=("faiss", "matrix"),
        help="Exact top-k backend. 'faiss' uses IndexFlatL2; 'matrix' uses torch matmul + topk and does not build a FAISS index.",
    )
    parser.add_argument(
        "--recall-cfg",
        default="",
        help="Optional recall threshold config name. Use 'per_scene' to resolve by scene, or pass cfg name such as cfg03. Empty keeps legacy thresholds.",
    )
    parser.add_argument(
        "--recall-cfg-yaml",
        default="trainer_depends/configs/stage3_recall_thresholds.yaml",
        help="YAML file containing recall threshold configs and optional per_scene mapping.",
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


def _resolve_stage1_ckpt(script_args, opt):
    if script_args.stage1_ckpt:
        ckpt_path = _resolve_cli_path(script_args.stage1_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
        return ckpt_path

    load_stage1_ckpt = getattr(opt, "load_stage1_ckpt", "")
    if load_stage1_ckpt:
        ckpt_path = _resolve_cli_path(load_stage1_ckpt)
        if os.path.exists(ckpt_path):
            return ckpt_path

    return ""


def _load_recall_cfg_yaml(path_text):
    path_abs = _resolve_cli_path(path_text)
    if not os.path.isfile(path_abs):
        raise FileNotFoundError(f"Recall cfg yaml not found: {path_abs}")

    import yaml

    with open(path_abs, "r", encoding="utf-8") as f:
        cfg_root = yaml.safe_load(f) or {}
    if not isinstance(cfg_root, dict):
        raise TypeError(f"Recall cfg yaml must be a dict: {path_abs}")
    return cfg_root, path_abs


def _resolve_scene_name_for_recall(trainer):
    scene_name = str(getattr(getattr(trainer, "sat_dataset", None), "name", "") or "").strip()
    if scene_name:
        return scene_name

    scene_name = str(getattr(trainer.opt, "selected_scene_name", "") or "").strip()
    if scene_name:
        return scene_name

    scenes_setting = getattr(trainer.opt, "scenes_setting", None)
    if isinstance(scenes_setting, dict):
        scene_name = str(scenes_setting.get("selected_scene_name", "") or "").strip()
        if scene_name:
            return scene_name
        scenes = scenes_setting.get("scenes", [])
        if scenes and isinstance(scenes[0], dict):
            return str(scenes[0].get("name", "") or "").strip()
    return ""


def _build_stage2_eval_thresh_cfg(sat_dataset, raw_cfg):
    raw_cfg = dict(raw_cfg)
    eval_thresh_cfg = {}

    if raw_cfg.get("dist_th_meter", None) is not None:
        if not hasattr(sat_dataset, "halfimg_radius_meter") or not hasattr(sat_dataset, "halfimg_radius_nrc"):
            raise ValueError("dist_th_meter requires sat_dataset.halfimg_radius_meter and sat_dataset.halfimg_radius_nrc")
        meter2nrc = float(sat_dataset.halfimg_radius_nrc) / max(float(sat_dataset.halfimg_radius_meter), 1e-8)
        dist_th = float(raw_cfg["dist_th_meter"]) * meter2nrc
        eval_thresh_cfg["dist_th"] = float(dist_th)
        eval_thresh_cfg["dist_lambda"] = float(dist_th) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
    elif raw_cfg.get("dist_th", None) is not None:
        dist_th = float(raw_cfg["dist_th"])
        eval_thresh_cfg["dist_th"] = dist_th
        if hasattr(sat_dataset, "halfimg_radius_nrc"):
            eval_thresh_cfg["dist_lambda"] = dist_th / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
    elif raw_cfg.get("dist_lambda", None) is not None:
        eval_thresh_cfg["dist_lambda"] = float(raw_cfg["dist_lambda"])

    if raw_cfg.get("rot_th_deg", None) is not None:
        eval_thresh_cfg["rot_th_deg"] = float(raw_cfg["rot_th_deg"])
    elif raw_cfg.get("rot_th", None) is not None:
        eval_thresh_cfg["rot_th"] = float(raw_cfg["rot_th"])

    if raw_cfg.get("scale_ratio_th", None) is not None:
        eval_thresh_cfg["scale_ratio_th"] = float(raw_cfg["scale_ratio_th"])
    elif raw_cfg.get("scale_th", None) is not None:
        eval_thresh_cfg["scale_th"] = float(raw_cfg["scale_th"])

    return eval_thresh_cfg


def _resolve_recall_eval_thresh_cfg(trainer, script_args):
    selector = str(getattr(script_args, "recall_cfg", "") or "").strip()
    if not selector or selector.lower() in {"none", "off", "legacy"}:
        return None

    trainer._ensure_stage2_eval_runtime()
    cfg_root, cfg_yaml_abs = _load_recall_cfg_yaml(script_args.recall_cfg_yaml)
    configs = cfg_root.get("configs", {})
    if not isinstance(configs, dict) or not configs:
        raise KeyError(f"Recall cfg yaml has no non-empty 'configs': {cfg_yaml_abs}")

    scene_name = _resolve_scene_name_for_recall(trainer)
    if selector.lower() in {"per_scene", "scene", "auto"}:
        per_scene = cfg_root.get("per_scene", {})
        if not isinstance(per_scene, dict):
            raise TypeError(f"Recall cfg 'per_scene' must be a dict: {cfg_yaml_abs}")
        cfg_name = str(per_scene.get(scene_name, cfg_root.get("default", ""))).strip()
    elif selector.lower() == "default":
        cfg_name = str(cfg_root.get("default", "")).strip()
    else:
        cfg_name = selector

    if cfg_name not in configs:
        available = ", ".join(sorted(str(k) for k in configs.keys()))
        raise KeyError(
            f"Recall cfg '{cfg_name}' not found for selector='{selector}', scene='{scene_name}'. "
            f"Available configs: {available}"
        )

    raw_cfg = configs[cfg_name]
    if not isinstance(raw_cfg, dict):
        raise TypeError(f"Recall cfg '{cfg_name}' must be a dict, got {type(raw_cfg).__name__}")

    eval_thresh_cfg = _build_stage2_eval_thresh_cfg(trainer.sat_dataset, raw_cfg)
    print(
        "[Stage2-RecallCfg] "
        f"yaml={cfg_yaml_abs} selector={selector} scene={scene_name} "
        f"resolved={cfg_name} thresholds={eval_thresh_cfg}"
    )
    return eval_thresh_cfg


def _resolve_mode_defaults(mode, n_bins_4d=None):
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
    if mode == "n_bins_4d":
        n_bins = _parse_n_bins_4d(n_bins_4d)
        if n_bins is None:
            raise ValueError("--n-bins-4d must be provided when --mode n_bins_4d.")
        has_rot = int(n_bins[2]) > 1
        has_scale = int(n_bins[3]) > 1
        if has_rot and has_scale:
            report_title = "Stage2 n_bins_4d RC + Rot + Scale Localization"
        elif has_rot:
            report_title = "Stage2 n_bins_4d RC + Rot Localization"
        elif has_scale:
            report_title = "Stage2 n_bins_4d RC + Scale Localization"
        else:
            report_title = "Stage2 n_bins_4d RC Localization"
        return {
            "query_rot2uniform": not has_rot,
            "query_scale2uniform": False,
            "report_rot_error": has_rot,
            "report_scale_error": has_scale,
            "report_title": report_title,
        }
    return defaults[mode]


def _build_layout_cfg(script_args):
    from trainers.util_stage2_gallery_manager import Stage2ReferenceGalleryLayoutConfig

    kwargs = {
        "mode": script_args.mode,
        "n_bins_4d": _parse_n_bins_4d(script_args.n_bins_4d),
        "overlap": script_args.overlap,
        "fixed_rot": script_args.fixed_rot,
        "fixed_scale": script_args.fixed_scale,
        "delta_rot_deg": script_args.delta_rot_deg,
        "n_scales": script_args.n_scales,
        "scale_mode": script_args.scale_mode,
    }
    if script_args.mode == "n_bins_4d" and kwargs["n_bins_4d"] is None:
        raise ValueError("--n-bins-4d must be provided when --mode n_bins_4d.")
    return Stage2ReferenceGalleryLayoutConfig(**kwargs)


def _build_eval_cfg(trainer, script_args, eval_thresh_cfg=None):
    mode_defaults = _resolve_mode_defaults(script_args.mode, script_args.n_bins_4d)
    batch_size = script_args.batch_size
    if batch_size is None:
        batch_size = int(getattr(trainer.opt, "batchsize_uav", 32))

    num_workers = script_args.num_workers_eval
    if num_workers is None:
        num_workers = int(getattr(trainer.opt, "num_worker_eval", 0))

    kwargs = dict(
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
        search_backend=str(script_args.search_backend),
    )
    if eval_thresh_cfg is not None:
        kwargs["eval_thresh_cfg"] = eval_thresh_cfg
    return trainer._make_stage2_retrieval_eval_cfg(**kwargs)


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
            "search_backend": str(getattr(eval_cfg, "search_backend", "faiss")),
            "k_values": [int(k) for k in eval_cfg.k_values],
            "max_queries": None if eval_cfg.max_queries is None else int(eval_cfg.max_queries),
        },
        "thresholds": eval_res.get("thresholds", {}),
        "report_meta": eval_res.get("report_meta", {}),
        "progressive_acc_metrics": eval_res.get("progressive_acc_metrics", {}),
        "progressive_acc_metric_sources": eval_res.get("progressive_acc_metric_sources", {}),
        "progressive_error_metrics": eval_res.get("progressive_error_metrics", {}),
        "progressive_error_metric_sources": eval_res.get("progressive_error_metric_sources", {}),
        "legacy_acc_metrics_source": eval_res.get("legacy_acc_metrics_source", None),
        "recall@k": {str(int(k)): float(v) for k, v in eval_res["recall@k"].items()},
        "error_rc_norm": float(eval_res["error_rc_norm"]),
        "error_rc_meter": float(eval_res["error_rc_meter"]),
        "search_backend": str(eval_res.get("search_backend", getattr(eval_cfg, "search_backend", "faiss"))),
        "retrieval_search_time_sec_total": float(eval_res.get("retrieval_search_time_sec_total", 0.0)),
        "retrieval_search_num_queries": int(eval_res.get("retrieval_search_num_queries", eval_res["n_queries"])),
        "retrieval_search_avg_sec_per_query": float(eval_res.get("retrieval_search_avg_sec_per_query", 0.0)),
        "retrieval_search_avg_ms_per_query": float(eval_res.get("retrieval_search_avg_ms_per_query", 0.0)),
        "retrieval_search_top_k": int(eval_res.get("retrieval_search_top_k", max(int(k) for k in eval_cfg.k_values))),
        "retrieval_search_gallery_size": int(
            eval_res.get(
                "retrieval_search_gallery_size",
                eval_res.get("runtime_gallery_summary", {}).get("n_points", 0),
            )
        ),
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
    print(
        "[Stage2 Fitness] retrieval_search({backend}): total={total:.6f}s | "
        "avg={avg:.6f}ms/query | top_k={top_k} | gallery_size={gallery_size}".format(
            backend=summary["search_backend"],
            total=summary["retrieval_search_time_sec_total"],
            avg=summary["retrieval_search_avg_ms_per_query"],
            top_k=summary["retrieval_search_top_k"],
            gallery_size=summary["retrieval_search_gallery_size"],
        )
    )


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

    opt.load_stage1_ckpt = _resolve_stage1_ckpt(script_args=script_args, opt=opt)
    opt.load2test = _resolve_stage2_ckpt(script_args, opt)

    stage1_ckpt_for_test = getattr(opt, "load_stage1_ckpt", "")
    opt.load_stage1_ckpt = ""
    trainer = GridHashFitTrainer(opt=opt)
    trainer.opt.load_stage1_ckpt = stage1_ckpt_for_test
    trainer._load_checkpoints_for_test()

    layout_cfg = _build_layout_cfg(script_args)
    eval_thresh_cfg = _resolve_recall_eval_thresh_cfg(trainer, script_args)
    eval_cfg = _build_eval_cfg(trainer, script_args, eval_thresh_cfg=eval_thresh_cfg)
    feature_cfg = trainer._make_stage2_gallery_feature_cfg()
    feature_cfg.build_faiss = str(script_args.search_backend).strip().lower() == "faiss"
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

# python /home/data/zwk/pyproj_neuloc_v0/scripts/eval_stage2_fitness.py \
#     --p_yaml /home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_visloc03_interval82_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/opts.yaml \
#     --stage2-ckpt /home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_visloc03_interval82_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/epoch960_R1=85_MeanErr192m_MidErr76m.pth \
#     --mode rc_rot_scale \
#     --overlap 0.375 \
#     --delta-rot-deg 10 \
#     --n-scales 4 \
#     --scale-mode linear \
#     --cache-gallery \
#     --gallery-name-prefix visloc03_interval82
