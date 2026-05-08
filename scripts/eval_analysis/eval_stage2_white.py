#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys


def _maybe_reexec_with_env_lib():
    """Ensure the conda env lib dir is first in LD_LIBRARY_PATH before heavy imports."""
    if os.environ.get("_EVAL_STAGE2_WHITE_BOOTSTRAPPED") == "1":
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
    env["_EVAL_STAGE2_WHITE_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else env_lib
    os.execve(python_bin, [python_bin] + sys.argv, env)


_maybe_reexec_with_env_lib()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import faiss
import torch
import torch.nn.functional as TF
import tqdm

from trainer_depends.config.parser import get_parse
from trainers.stage2_INGP import GridHashFitTrainer
from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords, print_progressive_topk_eval_results
from trainers.util_stage2_gallery_manager import (
    Stage2ReferenceGalleryBank,
    Stage2ReferenceGalleryFeatureConfig,
    Stage2ReferenceGalleryLayoutConfig,
)
from trainers.util_stage2_retrieval_evaluator import Stage2RetrievalEvaluator


DEFAULT_K_VALUES = (1, 5, 10, 20, 50, 256, 512, 1024)


def _parse_k_values(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("k_values cannot be empty.")
    return tuple(values)


def _resolve_cli_path(path_text):
    if not path_text:
        return ""
    if os.path.isabs(path_text):
        return path_text
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_text))


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


def _default_output_dir(stage2_ckpt, use_train_uav):
    ckpt_dir = os.path.dirname(os.path.abspath(stage2_ckpt))
    ckpt_tag = os.path.splitext(os.path.basename(stage2_ckpt))[0]
    split_tag = "trainq" if use_train_uav else "testq"
    return os.path.join(
        ckpt_dir,
        f"stage2_white_{ckpt_tag}_overlap000_rot018_scale1_{split_tag}",
    )


def _default_query_output_dir(stage1_ckpt, stage2_white_output_dir):
    stage1_dir = os.path.dirname(os.path.abspath(stage1_ckpt))
    stage2_white_tag = os.path.basename(os.path.abspath(stage2_white_output_dir))
    return os.path.join(stage1_dir, f"stage1_query_raw_from_{stage2_white_tag}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build Stage-2 rc_rot gallery, cache raw query feats, and evaluate raw vs whitened retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--p_yaml", required=True, help="Stage-2 base YAML or experiment opts.yaml.")
    parser.add_argument("--stage2-ckpt", default="", help="Stage-2 checkpoint path. If omitted, resolve automatically.")
    parser.add_argument("--stage1-ckpt", default="", help="Optional Stage-1 checkpoint override.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output dir. Default is inside the stage2 ckpt dir.")
    parser.add_argument("--use-train-uav", action="store_true", help="Use the train UAV split instead of the test split.")
    parser.add_argument("--batch-size", type=int, default=None, help="Query batch size. Default uses trainer batchsize_uav.")
    parser.add_argument("--num-workers-eval", type=int, default=0, help="Evaluation DataLoader workers.")
    parser.add_argument("--k-values", type=str, default=",".join(str(v) for v in DEFAULT_K_VALUES), help="Comma-separated recall@k.")
    parser.add_argument("--dist-lambda", type=float, default=0.5, help="Distance threshold as sat_dataset.halfimg_radius_nrc * dist_lambda.")
    parser.add_argument("--rot-th-deg", type=float, default=11.0, help="Rotation threshold in degrees for dist+rot recall.")
    parser.add_argument("--whiten-eps", type=float, default=1e-6, help="Relative epsilon multiplier for covariance whitening.")
    parser.add_argument("--rebuild-gallery", action="store_true", help="Force rebuild raw gallery coords/features.")
    parser.add_argument("--reextract-query", action="store_true", help="Force re-extract raw query feats.")
    parser.add_argument("--refit-whitening", action="store_true", help="Force refit whitening stats and recompute whitened feats.")
    return parser


def _build_trainer(script_args, remaining_argv):
    parser_argv = [sys.argv[0], "--p_yaml", script_args.p_yaml] + list(remaining_argv)
    sys.argv = parser_argv
    opt = get_parse(print_summary=False)

    if script_args.stage1_ckpt:
        opt.load_stage1_ckpt = _resolve_cli_path(script_args.stage1_ckpt)
        if not os.path.exists(opt.load_stage1_ckpt):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {opt.load_stage1_ckpt}")

    stage2_ckpt = _resolve_stage2_ckpt(script_args, opt)
    opt.load2test = stage2_ckpt

    stage1_ckpt_for_test = getattr(opt, "load_stage1_ckpt", "")
    opt.load_stage1_ckpt = ""
    trainer = GridHashFitTrainer(opt=opt)
    trainer.opt.load_stage1_ckpt = stage1_ckpt_for_test
    trainer._load_checkpoints_for_test()
    return trainer, stage2_ckpt


def _load_or_build_gallery(trainer, output_dir, rebuild_gallery):
    gallery_dir = os.path.join(output_dir, "gallery_raw")
    coords_path = os.path.join(gallery_dir, "coords_gallery.pt")
    feats_path = os.path.join(gallery_dir, "feats_gallery.pt")

    gallery_bank = Stage2ReferenceGalleryBank(
        sat_dataset=trainer.sat_dataset,
        trainer=trainer,
    )

    if (not rebuild_gallery) and os.path.exists(coords_path) and os.path.exists(feats_path):
        gallery_bank = Stage2ReferenceGalleryBank.load(
            gallery_dir,
            sat_dataset=trainer.sat_dataset,
            trainer=trainer,
            build_faiss=False,
        )
        return gallery_bank, gallery_dir, False

    layout_cfg = Stage2ReferenceGalleryLayoutConfig(
        mode="rc_rot",
        overlap=0.0,
        delta_rot_deg=20.0,
        n_scales=1,
    )
    feature_cfg = Stage2ReferenceGalleryFeatureConfig(
        chunk_size_coords=512,
        normalize_feats=True,
        build_faiss=False,
        show_progress=True,
    )
    gallery_bank.build_coords(layout_cfg)
    gallery_bank.build_features(feature_cfg)
    gallery_bank.save(gallery_dir, save_feats=True, save_meta=True)
    return gallery_bank, gallery_dir, True


def _extract_queries(trainer, use_train_uav, batch_size, num_workers):
    dummy_bank = Stage2ReferenceGalleryBank(sat_dataset=trainer.sat_dataset, trainer=trainer)
    dummy_bank.meta = {
        "scene_name": getattr(trainer.sat_dataset, "name", None),
        "n_points": 0,
        "mode": "query_only",
        "n_rot": None,
        "n_scale": None,
    }
    evaluator = Stage2RetrievalEvaluator(trainer=trainer, gallery_bank=dummy_bank, logger=trainer.logger)
    eval_cfg = trainer._make_stage2_retrieval_eval_cfg(
        use_train_uav=bool(use_train_uav),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        query_rot2uniform=False,
        query_scale2uniform=False,
        k_values=DEFAULT_K_VALUES,
        print_results=False,
        report_rot_error=True,
        report_scale_error=False,
    )
    scene_name, sat_dataset, uav_dataset = evaluator._resolve_runtime_datasets(eval_cfg)
    dataloader = evaluator._make_uav_dataloader(
        uav_dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )

    feats_all = []
    coords_all = []
    with torch.no_grad():
        iterator = tqdm.tqdm(
            dataloader,
            desc=f"[Stage2 White Query] {scene_name or 'unknown'}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                uavimgs, coords_uav = batch[0], batch[1]
            else:
                uavimgs, coords_uav = batch
            uavimgs = uavimgs.to(trainer.device)
            coords_uav = coords_uav.to(trainer.device)
            feats_q = evaluator._extract_query_feats(uavimgs)
            feats_all.append(feats_q.detach().cpu())
            coords_all.append(coords_uav.detach().cpu())

    query_feats = torch.cat(feats_all, dim=0)
    query_coords = torch.cat(coords_all, dim=0)
    return evaluator, sat_dataset, scene_name, query_feats, query_coords


def _load_or_extract_queries(trainer, query_dir, use_train_uav, batch_size, num_workers, reextract_query):
    feats_path = os.path.join(query_dir, "query_feats.pt")
    coords_path = os.path.join(query_dir, "query_coords.pt")
    meta_path = os.path.join(query_dir, "query_meta.json")

    if (not reextract_query) and os.path.exists(feats_path) and os.path.exists(coords_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if bool(meta.get("use_train_uav", False)) == bool(use_train_uav):
            query_feats = torch.load(feats_path, map_location="cpu")
            query_coords = torch.load(coords_path, map_location="cpu")
            dummy_bank = Stage2ReferenceGalleryBank(sat_dataset=trainer.sat_dataset, trainer=trainer)
            dummy_bank.meta = {
                "scene_name": meta.get("scene_name", getattr(trainer.sat_dataset, "name", None)),
                "n_points": 0,
                "mode": "query_only",
                "n_rot": None,
                "n_scale": None,
            }
            evaluator = Stage2RetrievalEvaluator(trainer=trainer, gallery_bank=dummy_bank, logger=trainer.logger)
            scene_name = meta.get("scene_name", getattr(trainer.sat_dataset, "name", None))
            sat_dataset = trainer.sat_dataset
            return evaluator, sat_dataset, scene_name, query_feats, query_coords, query_dir, False

    evaluator, sat_dataset, scene_name, query_feats, query_coords = _extract_queries(
        trainer=trainer,
        use_train_uav=use_train_uav,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    os.makedirs(query_dir, exist_ok=True)
    torch.save(query_feats, feats_path)
    torch.save(query_coords, coords_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "use_train_uav": bool(use_train_uav),
                "n_queries": int(query_coords.shape[0]),
                "feat_dim": int(query_feats.shape[1]),
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return evaluator, sat_dataset, scene_name, query_feats, query_coords, query_dir, True


def _fit_zca_whitening(feats_gallery, rel_eps):
    feats_gallery = feats_gallery.to(torch.float32)
    mu = feats_gallery.mean(dim=0, keepdim=True)
    centered = feats_gallery - mu
    denom = max(int(centered.shape[0]) - 1, 1)
    cov = centered.T @ centered / float(denom)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    max_eig = float(eigvals.max().item()) if eigvals.numel() > 0 else 0.0
    eps = float(rel_eps) * max(max_eig, 1.0)
    inv_sqrt = torch.rsqrt(eigvals + eps)
    whitening = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.T
    return {
        "mu": mu.cpu(),
        "whitening": whitening.cpu(),
        "eigvals": eigvals.cpu(),
        "eps": float(eps),
        "max_eig": float(max_eig),
    }


def _apply_whitening(feats, whitening_stats):
    feats = feats.to(torch.float32)
    mu = whitening_stats["mu"].to(feats.device)
    whitening = whitening_stats["whitening"].to(feats.device)
    feats_white = (feats - mu) @ whitening
    return TF.normalize(feats_white, dim=-1).cpu()


def _load_or_fit_whitening(output_dir, feats_gallery, query_feats, rel_eps, refit_whitening):
    white_dir = os.path.join(output_dir, "white")
    stats_path = os.path.join(white_dir, "whitening_stats.pt")
    gallery_white_path = os.path.join(white_dir, "feats_gallery_white.pt")
    query_white_path = os.path.join(white_dir, "query_feats_white.pt")

    if (
        (not refit_whitening)
        and os.path.exists(stats_path)
        and os.path.exists(gallery_white_path)
        and os.path.exists(query_white_path)
    ):
        whitening_stats = torch.load(stats_path, map_location="cpu")
        feats_gallery_white = torch.load(gallery_white_path, map_location="cpu")
        query_feats_white = torch.load(query_white_path, map_location="cpu")
        return whitening_stats, feats_gallery_white, query_feats_white, white_dir, False

    os.makedirs(white_dir, exist_ok=True)
    whitening_stats = _fit_zca_whitening(feats_gallery=feats_gallery, rel_eps=rel_eps)
    feats_gallery_white = _apply_whitening(feats_gallery, whitening_stats)
    query_feats_white = _apply_whitening(query_feats, whitening_stats)
    torch.save(whitening_stats, stats_path)
    torch.save(feats_gallery_white, gallery_white_path)
    torch.save(query_feats_white, query_white_path)
    return whitening_stats, feats_gallery_white, query_feats_white, white_dir, True


def _search_topk(feats_gallery, feats_query, top_k):
    index = faiss.IndexFlatL2(int(feats_gallery.shape[1]))
    index.add(feats_gallery.numpy())
    dists, indices = index.search(feats_query.numpy(), int(top_k))
    return dists, indices


def _build_eval_result(name, coords_gallery, coords_pred_idx, coords_gt, sat_dataset, thresholds, k_values):
    coords_topk = coords_gallery[torch.from_numpy(coords_pred_idx).long()]
    metrics, shared_errors = compute_progressive_topk_acc_from_coords(
        coords_topk,
        coords_gt,
        dist_th=thresholds["norm_dist"],
        rot_th_deg=thresholds["rot"],
        scale_ratio_th=thresholds["scale_ratio"],
        k_values=k_values,
    )

    top1_dist = torch.norm(coords_topk[:, 0, :2] - coords_gt[:, :2], p=2, dim=-1)
    rot_diff_rad = torch.abs(coords_topk[:, 0, 2] - coords_gt[:, 2])
    rot_errors_rad = torch.minimum(rot_diff_rad, 2 * torch.pi - rot_diff_rad)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)
    pred_scale = coords_topk[:, 0, 3].clamp(min=1e-6)
    gt_scale = coords_gt[:, 3].clamp(min=1e-6)
    scale_ratio = torch.maximum(pred_scale / gt_scale, gt_scale / pred_scale)
    dist_meter = float(sat_dataset.halfimg_radius_meter) * top1_dist / float(sat_dataset.halfimg_radius_nrc)

    recall_at_k = {str(int(k)): float(metrics.get(f"top{k}_acc", 0.0)) / 100.0 for k in k_values}
    return {
        "name": name,
        "n_queries": int(coords_gt.shape[0]),
        "metrics": metrics,
        "shared_errors": shared_errors,
        "recall@k": recall_at_k,
        "error_rc_norm": float(top1_dist.mean().item()),
        "error_rc_norm_median": float(torch.median(top1_dist).item()),
        "error_rc_meter": float(dist_meter.mean().item()),
        "error_rc_meter_median": float(torch.median(dist_meter).item()),
        "error_rot_deg": float(rot_errors_deg.mean().item()),
        "error_rot_deg_median": float(torch.median(rot_errors_deg).item()),
        "error_scale_ratio": float(scale_ratio.mean().item()),
        "error_scale_ratio_median": float(torch.median(scale_ratio).item()),
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_report(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = _build_arg_parser()
    script_args, remaining_argv = parser.parse_known_args(argv)

    trainer, stage2_ckpt = _build_trainer(script_args, remaining_argv)
    trainer._ensure_stage2_eval_runtime()
    batch_size = int(script_args.batch_size or getattr(trainer.opt, "batchsize_uav", 32))
    num_workers = int(script_args.num_workers_eval)
    k_values = _parse_k_values(script_args.k_values)

    output_dir = _resolve_cli_path(script_args.output_dir) if script_args.output_dir else _default_output_dir(
        stage2_ckpt=stage2_ckpt,
        use_train_uav=bool(script_args.use_train_uav),
    )
    os.makedirs(output_dir, exist_ok=True)
    trainer._save_gallery_stage2_provenance(output_dir, stage2_ckpt, load2test=stage2_ckpt)
    stage1_ckpt = trainer._get_stage1_checkpoint_path(stage2_ckpt)
    query_output_dir = _default_query_output_dir(stage1_ckpt, output_dir)

    gallery_bank, gallery_dir, gallery_rebuilt = _load_or_build_gallery(
        trainer=trainer,
        output_dir=output_dir,
        rebuild_gallery=bool(script_args.rebuild_gallery),
    )
    evaluator, sat_dataset, scene_name, query_feats, query_coords, query_dir, query_rebuilt = _load_or_extract_queries(
        trainer=trainer,
        query_dir=query_output_dir,
        use_train_uav=bool(script_args.use_train_uav),
        batch_size=batch_size,
        num_workers=num_workers,
        reextract_query=bool(script_args.reextract_query),
    )

    whitening_stats, feats_gallery_white, query_feats_white, white_dir, white_refit = _load_or_fit_whitening(
        output_dir=output_dir,
        feats_gallery=gallery_bank.feats_gallery,
        query_feats=query_feats,
        rel_eps=float(script_args.whiten_eps),
        refit_whitening=bool(script_args.refit_whitening),
    )

    eval_cfg = trainer._make_stage2_retrieval_eval_cfg(
        use_train_uav=bool(script_args.use_train_uav),
        batch_size=batch_size,
        num_workers=num_workers,
        query_rot2uniform=False,
        query_scale2uniform=False,
        k_values=k_values,
        dist_lambda=float(script_args.dist_lambda),
        rot_th_deg=float(script_args.rot_th_deg),
        scale_ratio_th=None,
        print_results=False,
        report_title="Stage2 White Eval",
        report_rot_error=True,
        report_scale_error=False,
    )
    thresholds = evaluator._resolve_thresholds(sat_dataset, eval_cfg)
    top_k = min(max(int(k) for k in k_values), int(gallery_bank.coords_gallery.shape[0]))

    _, idx_raw = _search_topk(gallery_bank.feats_gallery, query_feats, top_k=top_k)
    _, idx_white = _search_topk(feats_gallery_white, query_feats_white, top_k=top_k)

    raw_res = _build_eval_result(
        name="raw",
        coords_gallery=gallery_bank.coords_gallery,
        coords_pred_idx=idx_raw,
        coords_gt=query_coords,
        sat_dataset=sat_dataset,
        thresholds=thresholds,
        k_values=k_values,
    )
    white_res = _build_eval_result(
        name="white",
        coords_gallery=gallery_bank.coords_gallery,
        coords_pred_idx=idx_white,
        coords_gt=query_coords,
        sat_dataset=sat_dataset,
        thresholds=thresholds,
        k_values=k_values,
    )

    print_progressive_topk_eval_results(
        raw_res["metrics"],
        raw_res["shared_errors"],
        thresholds,
        report_title=f"Stage2 White Eval | RAW [{scene_name}]",
        report_meta={"integrate_scale": False},
    )
    print_progressive_topk_eval_results(
        white_res["metrics"],
        white_res["shared_errors"],
        thresholds,
        report_title=f"Stage2 White Eval | WHITE [{scene_name}]",
        report_meta={"integrate_scale": False},
    )

    summary = {
        "scene_name": scene_name,
        "stage2_ckpt": stage2_ckpt,
        "stage1_ckpt": stage1_ckpt,
        "output_dir": output_dir,
        "query_split": "train" if script_args.use_train_uav else "test",
        "gallery_layout": {
            "mode": "rc_rot",
            "overlap": 0.0,
            "n_rot": 18,
            "delta_rot_deg": 20.0,
            "n_scale": 1,
        },
        "thresholds": thresholds,
        "k_values": [int(k) for k in k_values],
        "cache_status": {
            "gallery_rebuilt": bool(gallery_rebuilt),
            "query_rebuilt": bool(query_rebuilt),
            "whitening_refit": bool(white_refit),
        },
        "artifacts": {
            "gallery_dir": gallery_dir,
            "query_dir": query_dir,
            "white_dir": white_dir,
            "gallery_coords": os.path.join(gallery_dir, "coords_gallery.pt"),
            "gallery_feats_raw": os.path.join(gallery_dir, "feats_gallery.pt"),
            "query_feats_raw": os.path.join(query_dir, "query_feats.pt"),
            "query_coords": os.path.join(query_dir, "query_coords.pt"),
            "whitening_stats": os.path.join(white_dir, "whitening_stats.pt"),
            "gallery_feats_white": os.path.join(white_dir, "feats_gallery_white.pt"),
            "query_feats_white": os.path.join(white_dir, "query_feats_white.pt"),
        },
        "whitening": {
            "method": "zca",
            "fit_source": "gallery_only",
            "relative_eps": float(script_args.whiten_eps),
            "effective_eps": float(whitening_stats["eps"]),
            "max_eig": float(whitening_stats["max_eig"]),
        },
        "raw": raw_res,
        "white": white_res,
        "delta_white_minus_raw": {
            f"R@{k}": float(white_res["recall@k"][str(int(k))] - raw_res["recall@k"][str(int(k))])
            for k in k_values
        },
        "delta_error": {
            "error_rc_meter_median": float(white_res["error_rc_meter_median"] - raw_res["error_rc_meter_median"]),
            "error_rot_deg_median": float(white_res["error_rot_deg_median"] - raw_res["error_rot_deg_median"]),
            "error_scale_ratio_median": float(white_res["error_scale_ratio_median"] - raw_res["error_scale_ratio_median"]),
        },
    }

    summary_path = os.path.join(output_dir, "stage2_white_summary.json")
    _write_json(summary_path, summary)

    report_lines = [
        "[Stage2 White Summary]",
        f"scene_name: {scene_name}",
        f"stage2_ckpt: {stage2_ckpt}",
        f"query_split: {'train' if script_args.use_train_uav else 'test'}",
        f"gallery_dir: {gallery_dir}",
        f"query_dir: {query_dir}",
        f"white_dir: {white_dir}",
        f"n_queries: {raw_res['n_queries']}",
        f"thresholds: dist_lambda={script_args.dist_lambda}, rot_th_deg={script_args.rot_th_deg}",
        "",
        "[RAW]",
        f"recall: " + " | ".join(f"R@{k}={raw_res['recall@k'][str(int(k))] * 100.0:.3f}%" for k in k_values),
        f"error_rc_meter_median: {raw_res['error_rc_meter_median']:.3f}m",
        f"error_rot_deg_median: {raw_res['error_rot_deg_median']:.3f}",
        "",
        "[WHITE]",
        f"recall: " + " | ".join(f"R@{k}={white_res['recall@k'][str(int(k))] * 100.0:.3f}%" for k in k_values),
        f"error_rc_meter_median: {white_res['error_rc_meter_median']:.3f}m",
        f"error_rot_deg_median: {white_res['error_rot_deg_median']:.3f}",
        "",
        "[DELTA WHITE-RAW]",
        "recall_delta: " + " | ".join(
            f"R@{k}={summary['delta_white_minus_raw'][f'R@{k}'] * 100.0:+.3f}%"
            for k in k_values
        ),
        f"error_rc_meter_median_delta: {summary['delta_error']['error_rc_meter_median']:+.3f}m",
        f"error_rot_deg_median_delta: {summary['delta_error']['error_rot_deg_median']:+.3f}",
        "",
    ]
    report_path = os.path.join(output_dir, "stage2_white_report.txt")
    _write_report(report_path, report_lines)

    print(f"[Stage2 White] saved summary: {summary_path}")
    print(f"[Stage2 White] saved report: {report_path}")


if __name__ == "__main__":
    main()
