#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Response-field analyzer for Stage-1 / Stage-2 reference galleries.

This utility focuses on task-correctness style metrics over the full response
field of a query against a gallery bank:
  - delta_peak
  - gt_rank
  - psr_gt
  - peak_correct_rate

The analyzer works with either:
  - Stage1ReferenceGalleryBank
  - Stage2ReferenceGalleryBank

Typical usage:

    from scripts.eval_analysis.analyze_response_field import (
        ResponseFieldAnalysisConfig,
        ResponseFieldGalleryLayoutConfig,
        ResponseFieldAnalyzer,
    )

    analyzer = ResponseFieldAnalyzer(gallery_bank, trainer=trainer)
    result = analyzer.analyze(
        ResponseFieldAnalysisConfig(
            metric="l2",
            bank_mode="continuous_like",
            query_rot2uniform=True,
        )
    )

    print(result["summary"])

    result = run_stage2_response_field_analysis(
        p_yaml="gen_fm_exps/ckpts/your_stage2_exp/opts.yaml",
        stage2_ckpt="gen_fm_exps/ckpts/your_stage2_exp/epoch100.pth",
        gallery_layout_cfg=ResponseFieldGalleryLayoutConfig(
            overlap=0.5,
            include_rot=False,
            include_scale=False,
            fixed_rot=0.0,
            fixed_scale=1.0,
        ),
        analysis_cfg=ResponseFieldAnalysisConfig(
            query_rot2uniform=True,
            metric="l2",
        ),
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as TF

from trainer_depends.utils.util_uav_image_transform import warp_uav_imgs


def _maybe_reexec_with_env_lib():
    """Ensure the conda env lib dir is first in LD_LIBRARY_PATH before heavy imports."""
    if os.environ.get("_ANALYZE_RESPONSE_FIELD_BOOTSTRAPPED") == "1":
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
    env["_ANALYZE_RESPONSE_FIELD_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else env_lib
    os.execve(python_bin, [python_bin] + sys.argv, env)


_maybe_reexec_with_env_lib()


@dataclass
class ResponseFieldAnalysisConfig:
    use_train_uav: bool = False
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    query_rot2uniform: bool = False
    query_scale2uniform: bool = False
    max_queries: Optional[int] = None
    metric: str = "l2"
    bank_mode: str = "discrete"
    r_near: Optional[float] = None
    r_far: Optional[float] = None
    print_results: bool = True
    report_title: str = "Response Field Analysis"
    save_per_query_csv: str = ""
    save_summary_json: str = ""


@dataclass
class ResponseFieldGalleryLayoutConfig:
    overlap: float = 0.5
    include_rot: bool = False
    include_scale: bool = False
    fixed_rot: float = 0.0
    fixed_scale: Optional[float] = None
    delta_rot_deg: float = 10.0
    n_scales: int = 1
    scale_mode: str = "linear"


def _normalize_metric(metric: str) -> str:
    metric = str(metric).strip().lower()
    if metric in ("l2", "euclidean"):
        return "l2"
    if metric in ("cos", "cosine"):
        return "cosine"
    raise ValueError(f"Unsupported metric: {metric}")


def _normalize_scale_mode(scale_mode: str) -> str:
    scale_mode = str(scale_mode).strip().lower()
    if scale_mode not in ("linear", "log"):
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")
    return scale_mode


def _normalize_bank_mode(bank_mode: str) -> str:
    bank_mode = str(bank_mode).strip().lower()
    valid = ("discrete", "continuous_like")
    if bank_mode not in valid:
        raise ValueError(f"bank_mode must be one of {valid}, got {bank_mode}")
    return bank_mode


def _ensure_parent_dir(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def compute_response_scores(
    feats_query: torch.Tensor,
    feats_gallery: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    """
    Compute query-vs-gallery response scores where larger is always better.
    """
    metric = _normalize_metric(metric)
    feats_query = TF.normalize(feats_query, dim=-1)
    feats_gallery = TF.normalize(feats_gallery, dim=-1)

    if metric == "cosine":
        return feats_query @ feats_gallery.transpose(0, 1)

    dists = torch.cdist(feats_query, feats_gallery, p=2)
    return -dists


def compute_pair_scores(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    """
    Compute aligned pair scores for two [B, D] tensors, one score per row.
    """
    metric = _normalize_metric(metric)
    feats_a = TF.normalize(feats_a, dim=-1)
    feats_b = TF.normalize(feats_b, dim=-1)

    if metric == "cosine":
        return torch.sum(feats_a * feats_b, dim=-1)

    return -torch.linalg.norm(feats_a - feats_b, dim=-1)


def compute_peak_metrics(
    scores: torch.Tensor,
    coords_gallery: torch.Tensor,
    coords_gt: torch.Tensor,
    r_near: float,
    r_far: float,
    exact_gt_scores: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-query response-field metrics.

    Args:
        scores: [B, N] response scores, larger is better.
        coords_gallery: [N, 4] gallery coords.
        coords_gt: [B, 4] query GT coords.
        r_near: radius of the positive neighborhood in rc space.
        r_far: radius outside which a point is considered a far distractor.
        exact_gt_scores: optional [B] exact GT response scores for
            continuous-like analysis.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}")
    if coords_gallery.ndim != 2 or coords_gallery.shape[-1] < 2:
        raise ValueError(f"coords_gallery must be [N, >=2], got {tuple(coords_gallery.shape)}")
    if coords_gt.ndim != 2 or coords_gt.shape[-1] < 2:
        raise ValueError(f"coords_gt must be [B, >=2], got {tuple(coords_gt.shape)}")
    if scores.shape[0] != coords_gt.shape[0] or scores.shape[1] != coords_gallery.shape[0]:
        raise ValueError(
            "Shape mismatch among scores / coords_gallery / coords_gt: "
            f"{tuple(scores.shape)}, {tuple(coords_gallery.shape)}, {tuple(coords_gt.shape)}"
        )
    if r_far < r_near:
        raise ValueError(f"r_far must be >= r_near, got r_near={r_near}, r_far={r_far}")

    coords_gallery_rc = coords_gallery[:, :2].to(scores.device, dtype=torch.float32)
    coords_gt_rc = coords_gt[:, :2].to(scores.device, dtype=torch.float32)
    dist_rc = torch.cdist(coords_gt_rc, coords_gallery_rc, p=2)

    batch_size = scores.shape[0]
    out = {
        "delta_peak": [],
        "gt_rank": [],
        "psr_gt": [],
        "is_peak_correct": [],
        "s_near": [],
        "s_near_discrete": [],
        "s_near_exact": [],
        "s_far_max": [],
        "mean_far": [],
        "std_far": [],
        "sampling_gap": [],
        "near_count": [],
        "far_count": [],
        "fallback_nearest_used": [],
    }

    for idx in range(batch_size):
        scores_i = scores[idx]
        dist_i = dist_rc[idx]

        near_mask = dist_i <= float(r_near)
        fallback_nearest_used = False
        if not torch.any(near_mask):
            near_mask = torch.zeros_like(dist_i, dtype=torch.bool)
            near_mask[torch.argmin(dist_i)] = True
            fallback_nearest_used = True

        far_mask = dist_i > float(r_far)
        if not torch.any(far_mask):
            far_mask = dist_i > float(r_near)
        if not torch.any(far_mask):
            far_mask = torch.ones_like(dist_i, dtype=torch.bool)
            far_mask[torch.argmin(dist_i)] = False
        if not torch.any(far_mask):
            far_mask = torch.ones_like(dist_i, dtype=torch.bool)

        s_near_discrete = torch.max(scores_i[near_mask])
        if exact_gt_scores is None:
            s_near_exact = torch.tensor(float("nan"), device=scores.device, dtype=scores.dtype)
            s_near = s_near_discrete
        else:
            s_near_exact = exact_gt_scores[idx]
            s_near = torch.maximum(s_near_discrete, s_near_exact)

        far_scores = scores_i[far_mask]
        s_far_max = torch.max(far_scores)
        mean_far = torch.mean(far_scores)
        std_far = torch.std(far_scores, unbiased=False).clamp_min(float(eps))

        delta_peak = s_near - s_far_max
        gt_rank = 1 + torch.sum(far_scores > s_near)
        psr_gt = (s_near - mean_far) / std_far
        is_peak_correct = delta_peak > 0
        sampling_gap = s_near - s_near_discrete

        out["delta_peak"].append(delta_peak)
        out["gt_rank"].append(gt_rank.to(torch.int64))
        out["psr_gt"].append(psr_gt)
        out["is_peak_correct"].append(is_peak_correct.to(torch.float32))
        out["s_near"].append(s_near)
        out["s_near_discrete"].append(s_near_discrete)
        out["s_near_exact"].append(s_near_exact)
        out["s_far_max"].append(s_far_max)
        out["mean_far"].append(mean_far)
        out["std_far"].append(std_far)
        out["sampling_gap"].append(sampling_gap)
        out["near_count"].append(torch.tensor(int(near_mask.sum().item()), device=scores.device, dtype=torch.int64))
        out["far_count"].append(torch.tensor(int(far_mask.sum().item()), device=scores.device, dtype=torch.int64))
        out["fallback_nearest_used"].append(
            torch.tensor(1.0 if fallback_nearest_used else 0.0, device=scores.device, dtype=torch.float32)
        )

    return {
        key: torch.stack(values, dim=0)
        for key, values in out.items()
    }


def summarize_peak_metrics(
    metric_tensors: Dict[str, torch.Tensor],
    cfg: ResponseFieldAnalysisConfig,
    scene_name: Optional[str] = None,
) -> Dict[str, object]:
    delta_peak = metric_tensors["delta_peak"].detach().cpu().to(torch.float32)
    gt_rank = metric_tensors["gt_rank"].detach().cpu().to(torch.int64)
    psr_gt = metric_tensors["psr_gt"].detach().cpu().to(torch.float32)
    is_peak_correct = metric_tensors["is_peak_correct"].detach().cpu().to(torch.float32)
    sampling_gap = metric_tensors["sampling_gap"].detach().cpu().to(torch.float32)
    fallback_nearest_used = metric_tensors["fallback_nearest_used"].detach().cpu().to(torch.float32)

    return {
        "scene_name": scene_name,
        "n_queries": int(delta_peak.numel()),
        "metric": _normalize_metric(cfg.metric),
        "bank_mode": _normalize_bank_mode(cfg.bank_mode),
        "query_rot2uniform": bool(cfg.query_rot2uniform),
        "query_scale2uniform": bool(cfg.query_scale2uniform),
        "r_near": None if cfg.r_near is None else float(cfg.r_near),
        "r_far": None if cfg.r_far is None else float(cfg.r_far),
        "peak_correct_rate": float(is_peak_correct.mean().item()),
        "mean_delta_peak": float(delta_peak.mean().item()),
        "median_delta_peak": float(delta_peak.median().item()),
        "mean_gt_rank": float(gt_rank.to(torch.float32).mean().item()),
        "median_gt_rank": float(gt_rank.to(torch.float32).median().item()),
        "gt@1": float((gt_rank <= 1).to(torch.float32).mean().item()),
        "gt@5": float((gt_rank <= 5).to(torch.float32).mean().item()),
        "gt@10": float((gt_rank <= 10).to(torch.float32).mean().item()),
        "mean_psr_gt": float(psr_gt.mean().item()),
        "median_psr_gt": float(psr_gt.median().item()),
        "mean_sampling_gap": float(sampling_gap.mean().item()),
        "fallback_nearest_rate": float(fallback_nearest_used.mean().item()),
    }


class ResponseFieldAnalyzer:
    """
    Full-response-field analyzer for a single gallery bank.
    """

    def __init__(self, gallery_bank, trainer=None, logger=None):
        self.gallery_bank = gallery_bank
        self.trainer = trainer or getattr(gallery_bank, "trainer", None)
        if self.trainer is None:
            raise ValueError("trainer is required either explicitly or via gallery_bank.trainer.")
        self.logger = logger or getattr(self.trainer, "logger", None)
        self.device = getattr(self.trainer, "device", torch.device("cpu"))

    def _log(self, msg: str) -> None:
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def _ensure_gallery_ready(self) -> None:
        if self.gallery_bank.coords_gallery is None:
            raise ValueError("gallery_bank.build_coords(...) must be called before response analysis.")
        if self.gallery_bank.feats_gallery is None:
            self.gallery_bank.build_features()

    def _resolve_runtime_datasets(self, cfg: ResponseFieldAnalysisConfig):
        if not hasattr(self.trainer, "sat_dataset"):
            self.trainer._init_datasets(create_train_loader=False)

        scene_name = self.gallery_bank.meta.get("scene_name", getattr(self.trainer.sat_dataset, "name", None))
        if (
            hasattr(self.trainer, "sat_datasets")
            and scene_name is not None
            and scene_name in self.trainer.sat_datasets
        ):
            sat_dataset = self.trainer.sat_datasets[scene_name]
            uav_dataset = (
                self.trainer.uav_datasets_train[scene_name]
                if cfg.use_train_uav else
                self.trainer.uav_datasets_test[scene_name]
            )
        else:
            sat_dataset = self.trainer.sat_dataset
            uav_dataset = self.trainer.uav_dataset_train if cfg.use_train_uav else self.trainer.uav_dataset_test
        return scene_name, sat_dataset, uav_dataset

    @staticmethod
    def _make_uav_dataloader(uav_dataset, batch_size: int, num_workers: int):
        return torch.utils.data.DataLoader(
            uav_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    def _prepare_query_batch(self, imgs, coords_uav, gallery_scale: float, cfg: ResponseFieldAnalysisConfig):
        if not cfg.query_rot2uniform and not cfg.query_scale2uniform:
            return imgs, coords_uav

        rot_align = -coords_uav[:, 2] if cfg.query_rot2uniform else None
        scale_f = None
        if cfg.query_scale2uniform:
            scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)

        imgs = warp_uav_imgs(imgs, rot_rad=rot_align, scale_f=scale_f)
        coords_uav = coords_uav.clone()
        if cfg.query_rot2uniform:
            coords_uav[:, 2] = 0
        if cfg.query_scale2uniform:
            coords_uav[:, 3] = gallery_scale
        return imgs, coords_uav

    def _extract_query_feats(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats_q = self.trainer._get_feats_fm_imgs(imgs)
            return TF.normalize(feats_q, dim=-1)

    def _extract_exact_gt_feats(self, coords_4d: torch.Tensor) -> torch.Tensor:
        coords_4d = coords_4d.to(self.device, dtype=torch.float32)

        if hasattr(self.trainer, "_extract_stage2_feats_from_coords_chunk"):
            return self.trainer._extract_stage2_feats_from_coords_chunk(coords_4d, normalize=True)

        if hasattr(self.gallery_bank, "_crop_gallery_images"):
            apply_rotation = bool(self.gallery_bank.meta.get("gallery_has_rot", False))
            satimgs = self.gallery_bank._crop_gallery_images(
                coords_chunk=coords_4d.detach().cpu(),
                apply_rotation=apply_rotation,
                chunk_size=int(coords_4d.shape[0]),
            )
            satimgs = satimgs.to(self.device)
        elif hasattr(self.gallery_bank.sat_dataset, "crop_satimg_by_4d_coords_fast"):
            satimgs = self.gallery_bank.sat_dataset.crop_satimg_by_4d_coords_fast(
                coords_4d.detach().cpu(),
                apply_rotation=bool(self.gallery_bank.meta.get("gallery_has_rot", False)),
                chunk_size=int(coords_4d.shape[0]),
            ).to(self.device)
        elif hasattr(self.gallery_bank.sat_dataset, "crop_satimg_by_4d_coords"):
            satimgs = self.gallery_bank.sat_dataset.crop_satimg_by_4d_coords(
                coords_4d.detach().cpu(),
                apply_rotation=bool(self.gallery_bank.meta.get("gallery_has_rot", False)),
            ).to(self.device)
        else:
            raise AttributeError(
                "Cannot build exact GT features: neither trainer Stage-2 coord extractor nor sat crop API is available."
            )

        with torch.no_grad():
            feats_gt = self.trainer._get_feats_fm_imgs(satimgs)
            return TF.normalize(feats_gt, dim=-1)

    def _enter_eval_mode(self):
        models_all = list(self.trainer.param2optimize.values()) + list(self.trainer.param2freeze.values())
        orig_modes = [m.training for m in models_all]
        for model in models_all:
            model.eval()
        return models_all, orig_modes

    @staticmethod
    def _restore_modes(models_all, orig_modes):
        for model, was_train in zip(models_all, orig_modes):
            model.train(was_train)

    def analyze(self, analysis_cfg=None) -> Dict[str, object]:
        self._ensure_gallery_ready()

        cfg = analysis_cfg if isinstance(analysis_cfg, ResponseFieldAnalysisConfig) else (
            ResponseFieldAnalysisConfig(**analysis_cfg) if analysis_cfg is not None else ResponseFieldAnalysisConfig()
        )
        cfg.metric = _normalize_metric(cfg.metric)
        cfg.bank_mode = _normalize_bank_mode(cfg.bank_mode)
        if cfg.batch_size is None:
            cfg.batch_size = int(getattr(self.trainer.opt, "batchsize_uav", 32))
        if cfg.num_workers is None:
            cfg.num_workers = int(getattr(self.trainer.opt, "num_worker_eval", 0))

        scene_name, sat_dataset, uav_dataset = self._resolve_runtime_datasets(cfg)
        dataloader = self._make_uav_dataloader(
            uav_dataset=uav_dataset,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )

        if cfg.r_near is None:
            cfg.r_near = float(sat_dataset.halfimg_radius_nrc)
        if cfg.r_far is None:
            cfg.r_far = float(cfg.r_near) * 1.01

        gallery_scale = float(
            self.gallery_bank.meta.get("gallery_scale_mean", getattr(sat_dataset, "satimgsize_scale_to_ref_m_mean", 1.0))
        )
        feats_gallery = TF.normalize(self.gallery_bank.feats_gallery.to(self.device, dtype=torch.float32), dim=-1)
        coords_gallery = self.gallery_bank.coords_gallery.to(self.device, dtype=torch.float32)

        metric_accum = {
            "delta_peak": [],
            "gt_rank": [],
            "psr_gt": [],
            "is_peak_correct": [],
            "s_near": [],
            "s_near_discrete": [],
            "s_near_exact": [],
            "s_far_max": [],
            "mean_far": [],
            "std_far": [],
            "sampling_gap": [],
            "near_count": [],
            "far_count": [],
            "fallback_nearest_used": [],
        }
        per_query_rows: List[Dict[str, object]] = []

        processed = 0
        models_all, orig_modes = self._enter_eval_mode()
        try:
            for batch in dataloader:
                if cfg.max_queries is not None and processed >= int(cfg.max_queries):
                    break

                if isinstance(batch, (list, tuple)):
                    imgs, coords_uav = batch[0], batch[1]
                else:
                    imgs, coords_uav = batch

                imgs = imgs.to(self.device)
                coords_uav = coords_uav.to(self.device, dtype=torch.float32)

                if cfg.max_queries is not None:
                    remain = int(cfg.max_queries) - processed
                    if imgs.shape[0] > remain:
                        imgs = imgs[:remain]
                        coords_uav = coords_uav[:remain]

                imgs, coords_uav = self._prepare_query_batch(imgs, coords_uav, gallery_scale, cfg)
                feats_q = self._extract_query_feats(imgs)
                scores = compute_response_scores(feats_q, feats_gallery, metric=cfg.metric)

                exact_gt_scores = None
                if cfg.bank_mode == "continuous_like":
                    feats_gt_exact = self._extract_exact_gt_feats(coords_uav)
                    exact_gt_scores = compute_pair_scores(feats_q, feats_gt_exact, metric=cfg.metric)

                batch_metrics = compute_peak_metrics(
                    scores=scores,
                    coords_gallery=coords_gallery,
                    coords_gt=coords_uav,
                    r_near=float(cfg.r_near),
                    r_far=float(cfg.r_far),
                    exact_gt_scores=exact_gt_scores,
                )

                batch_size_now = coords_uav.shape[0]
                for key, value in batch_metrics.items():
                    metric_accum[key].append(value.detach().cpu())

                for local_idx in range(batch_size_now):
                    global_query_idx = processed + local_idx
                    row = {
                        "query_index": int(global_query_idx),
                        "scene_name": scene_name,
                        "delta_peak": float(batch_metrics["delta_peak"][local_idx].item()),
                        "gt_rank": int(batch_metrics["gt_rank"][local_idx].item()),
                        "psr_gt": float(batch_metrics["psr_gt"][local_idx].item()),
                        "is_peak_correct": int(batch_metrics["is_peak_correct"][local_idx].item() > 0.5),
                        "s_near": float(batch_metrics["s_near"][local_idx].item()),
                        "s_near_discrete": float(batch_metrics["s_near_discrete"][local_idx].item()),
                        "s_near_exact": float(batch_metrics["s_near_exact"][local_idx].item()),
                        "s_far_max": float(batch_metrics["s_far_max"][local_idx].item()),
                        "mean_far": float(batch_metrics["mean_far"][local_idx].item()),
                        "std_far": float(batch_metrics["std_far"][local_idx].item()),
                        "sampling_gap": float(batch_metrics["sampling_gap"][local_idx].item()),
                        "near_count": int(batch_metrics["near_count"][local_idx].item()),
                        "far_count": int(batch_metrics["far_count"][local_idx].item()),
                        "fallback_nearest_used": int(batch_metrics["fallback_nearest_used"][local_idx].item() > 0.5),
                    }
                    per_query_rows.append(row)

                processed += batch_size_now
        finally:
            self._restore_modes(models_all, orig_modes)

        if not per_query_rows:
            raise ValueError("No valid queries were processed during response-field analysis.")

        metric_tensors = {
            key: torch.cat(value, dim=0)
            for key, value in metric_accum.items()
        }
        summary = summarize_peak_metrics(
            metric_tensors=metric_tensors,
            cfg=cfg,
            scene_name=scene_name,
        )
        runtime_gallery_summary = self.gallery_bank.summary()
        summary["runtime_gallery_summary"] = runtime_gallery_summary
        summary["gallery_overlap"] = runtime_gallery_summary.get("overlap", None)
        summary["gallery_n_bins_4d"] = runtime_gallery_summary.get("n_bins_4d", None)
        summary["gallery_shape"] = runtime_gallery_summary.get("gallery_shape", None)
        summary["analysis_cfg"] = asdict(cfg)

        result = {
            "summary": summary,
            "per_query": per_query_rows,
            "metric_tensors": metric_tensors,
        }

        if cfg.print_results:
            self._print_summary(summary, title=cfg.report_title)
        if cfg.save_per_query_csv:
            self._write_per_query_csv(cfg.save_per_query_csv, per_query_rows)
        if cfg.save_summary_json:
            self._write_summary_json(cfg.save_summary_json, summary)

        return result

    def _print_summary(self, summary: Dict[str, object], title: str) -> None:
        title_text = title
        if summary.get("scene_name"):
            title_text = f"{title} [{summary['scene_name']}]"
        self._log("")
        self._log(f"[{title_text}] N={int(summary['n_queries'])}")
        self._log(
            "[{title}] peak_correct_rate={pcr:.4f} | mean_delta_peak={mdp:.4f} | median_delta_peak={medp:.4f}".format(
                title=title_text,
                pcr=float(summary["peak_correct_rate"]),
                mdp=float(summary["mean_delta_peak"]),
                medp=float(summary["median_delta_peak"]),
            )
        )
        self._log(
            "[{title}] mean_gt_rank={mgr:.4f} | median_gt_rank={medr:.4f} | "
            "GT@1={gt1:.4f} | GT@5={gt5:.4f} | GT@10={gt10:.4f}".format(
                title=title_text,
                mgr=float(summary["mean_gt_rank"]),
                medr=float(summary["median_gt_rank"]),
                gt1=float(summary["gt@1"]),
                gt5=float(summary["gt@5"]),
                gt10=float(summary["gt@10"]),
            )
        )
        self._log(
            "[{title}] mean_psr_gt={mpsr:.4f} | median_psr_gt={medpsr:.4f} | "
            "mean_sampling_gap={msg:.4f}".format(
                title=title_text,
                mpsr=float(summary["mean_psr_gt"]),
                medpsr=float(summary["median_psr_gt"]),
                msg=float(summary["mean_sampling_gap"]),
            )
        )

    @staticmethod
    def _write_per_query_csv(path: str, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        _ensure_parent_dir(path)
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[ResponseFieldAnalyzer] saved per-query csv: {os.path.abspath(path)}")

    @staticmethod
    def _write_summary_json(path: str, summary: Dict[str, object]) -> None:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[ResponseFieldAnalyzer] saved summary json: {os.path.abspath(path)}")


def _resolve_cli_path(path_text: str) -> str:
    if not path_text:
        return ""
    if os.path.isabs(path_text):
        return path_text
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_text))


def _find_latest_epoch_ckpt(directory: str) -> Optional[str]:
    if not directory or not os.path.isdir(directory):
        return None

    best = None
    for name in os.listdir(directory):
        match = re.fullmatch(r"epoch(\d+)(?:.*)?\.pth", name)
        if not match:
            continue
        epoch = int(match.group(1))
        path = os.path.join(directory, name)
        if best is None or epoch > best[0] or (epoch == best[0] and "best" in name and "best" not in best[1]):
            best = (epoch, name, path)
    return None if best is None else best[2]


def _resolve_stage2_ckpt_path(p_yaml: str, stage2_ckpt: str, opt) -> str:
    if stage2_ckpt:
        ckpt_path = _resolve_cli_path(stage2_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")
        return ckpt_path

    p_yaml_abs = _resolve_cli_path(p_yaml)
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


def _normalize_gallery_layout_cfg(layout_cfg):
    cfg = layout_cfg if isinstance(layout_cfg, ResponseFieldGalleryLayoutConfig) else (
        ResponseFieldGalleryLayoutConfig(**layout_cfg)
    )
    cfg.overlap = float(cfg.overlap)
    if not (0.0 <= cfg.overlap < 1.0):
        raise ValueError("gallery overlap must be in [0, 1).")
    cfg.include_rot = bool(cfg.include_rot)
    cfg.include_scale = bool(cfg.include_scale)
    cfg.fixed_rot = float(cfg.fixed_rot)
    cfg.fixed_scale = None if cfg.fixed_scale is None else float(cfg.fixed_scale)
    cfg.delta_rot_deg = float(cfg.delta_rot_deg)
    if cfg.delta_rot_deg <= 0:
        raise ValueError("gallery delta_rot_deg must be > 0.")
    cfg.n_scales = int(cfg.n_scales)
    if cfg.n_scales <= 0:
        raise ValueError("gallery n_scales must be > 0.")
    cfg.scale_mode = _normalize_scale_mode(cfg.scale_mode)
    return cfg


def _layout_mode_from_cfg(layout_cfg: ResponseFieldGalleryLayoutConfig) -> str:
    if layout_cfg.include_rot and layout_cfg.include_scale:
        return "rc_rot_scale"
    if layout_cfg.include_rot:
        return "rc_rot"
    if layout_cfg.include_scale:
        return "rc_scale"
    return "rc"


def _build_stage2_layout_cfg(layout_cfg):
    from trainers.util_stage2_gallery_manager import Stage2ReferenceGalleryLayoutConfig

    cfg = _normalize_gallery_layout_cfg(layout_cfg)
    return Stage2ReferenceGalleryLayoutConfig(
        mode=_layout_mode_from_cfg(cfg),
        overlap=float(cfg.overlap),
        fixed_rot=float(cfg.fixed_rot),
        fixed_scale=cfg.fixed_scale,
        delta_rot_deg=float(cfg.delta_rot_deg),
        n_scales=int(cfg.n_scales),
        scale_mode=cfg.scale_mode,
    )


def _build_gallery_layout_cfg_from_args(args):
    include_rot = bool(getattr(args, "gallery_include_rot", False))
    include_scale = bool(getattr(args, "gallery_include_scale", False))

    return _normalize_gallery_layout_cfg(
        ResponseFieldGalleryLayoutConfig(
            overlap=float(args.gallery_overlap),
            include_rot=include_rot,
            include_scale=include_scale,
            fixed_rot=float(args.gallery_fixed_rot),
            fixed_scale=args.gallery_fixed_scale,
            delta_rot_deg=float(args.gallery_delta_rot_deg),
            n_scales=int(args.gallery_n_scales),
            scale_mode=args.gallery_scale_mode,
        )
    )


def _resolve_stage2_gallery_ckpt_tag(ckpt_path: str) -> Optional[str]:
    if not ckpt_path:
        return None
    ckpt_name = os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    return ckpt_name or None


def _resolve_gallery_cache_dir(
    trainer,
    layout_cfg: ResponseFieldGalleryLayoutConfig,
    gallery_root_dir: Optional[str],
    gallery_name_prefix: Optional[str],
    ckpt_path: str,
) -> str:
    scene_name = getattr(trainer.sat_dataset, "name", "default_scene")
    name_prefix = gallery_name_prefix or scene_name
    root_dir = gallery_root_dir or os.path.join(PROJECT_ROOT, "gen_fm_exps", "gallery_bank_stage2")

    mode = _layout_mode_from_cfg(layout_cfg)
    overlap_tag = f"overlap{int(round(float(layout_cfg.overlap) * 100.0)):03d}"
    layout_tags = [name_prefix, mode, overlap_tag]
    if layout_cfg.fixed_scale is not None:
        layout_tags.append(f"fixs{float(layout_cfg.fixed_scale):.3f}".replace(".", "p"))
    if (not layout_cfg.include_rot) and abs(float(layout_cfg.fixed_rot)) > 1e-6:
        layout_tags.append(f"fixr{float(layout_cfg.fixed_rot):.3f}".replace(".", "p"))
    if layout_cfg.include_rot:
        layout_tags.append(f"drot{float(layout_cfg.delta_rot_deg):g}".replace(".", "p"))
    if layout_cfg.include_scale:
        layout_tags.append(f"nscale{int(layout_cfg.n_scales)}")
    layout_tags.append(str(layout_cfg.scale_mode))

    base_dir = os.path.join(root_dir, "_".join(layout_tags))
    ckpt_tag = _resolve_stage2_gallery_ckpt_tag(ckpt_path)
    if ckpt_tag:
        return os.path.join(base_dir, ckpt_tag)
    return base_dir


def _build_stage2_feature_cfg():
    from trainers.util_stage2_gallery_manager import Stage2ReferenceGalleryFeatureConfig

    return Stage2ReferenceGalleryFeatureConfig(
        chunk_size_coords=512,
        normalize_feats=True,
        build_faiss=False,
        show_progress=True,
    )


def _build_or_load_stage2_gallery_bank(
    trainer,
    layout_cfg: ResponseFieldGalleryLayoutConfig,
    feature_cfg,
    cache_gallery: bool = False,
    gallery_root_dir: Optional[str] = None,
    gallery_name_prefix: Optional[str] = None,
    ckpt_path: str = "",
):
    from trainers.util_stage2_gallery_manager import Stage2ReferenceGalleryBank

    trainer._ensure_stage2_eval_runtime()
    stage2_layout_cfg = _build_stage2_layout_cfg(layout_cfg)
    gallery_save_dir = None
    if cache_gallery:
        gallery_save_dir = _resolve_gallery_cache_dir(
            trainer=trainer,
            layout_cfg=layout_cfg,
            gallery_root_dir=gallery_root_dir,
            gallery_name_prefix=gallery_name_prefix,
            ckpt_path=ckpt_path,
        )

    coords_path = None if gallery_save_dir is None else os.path.join(gallery_save_dir, "coords_gallery.pt")
    can_load = bool(cache_gallery and coords_path and os.path.exists(coords_path))

    models_all, orig_modes = trainer._enter_model_eval_mode()
    try:
        if can_load:
            gallery_bank = Stage2ReferenceGalleryBank.load(
                gallery_save_dir,
                sat_dataset=trainer.sat_dataset,
                trainer=trainer,
                build_faiss=False,
            )
            if gallery_bank.feats_gallery is None:
                gallery_bank.build_features(feature_cfg)
                gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
        else:
            gallery_bank = Stage2ReferenceGalleryBank(sat_dataset=trainer.sat_dataset, trainer=trainer)
            gallery_bank.build_coords(stage2_layout_cfg)
            gallery_bank.build_features(feature_cfg)
            if gallery_save_dir is not None:
                gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
    finally:
        trainer._restore_model_modes(models_all, orig_modes)

    return gallery_bank, stage2_layout_cfg


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze response-field metrics with explicit gallery layout settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--p_yaml", required=True, help="Stage-2 base YAML or experiment opts.yaml.")
    parser.add_argument("--stage2-ckpt", default="", help="Stage-2 checkpoint path.")
    parser.add_argument("--stage1-ckpt", default="", help="Optional Stage-1 checkpoint override.")
    parser.add_argument("--gallery-overlap", "--overlap", dest="gallery_overlap", type=float, default=0.5)
    parser.add_argument("--gallery-include-rot", action="store_true")#如果 --gallery-include-rot 没开，fixed_rot 生效
    parser.add_argument("--gallery-include-scale", action="store_true") #如果 --gallery-include-scale 没开，fixed_scale 生效
    parser.add_argument("--gallery-fixed-scale", "--fixed-scale", dest="gallery_fixed_scale", type=float, default=1.0)
    parser.add_argument("--gallery-fixed-rot", "--fixed-rot", dest="gallery_fixed_rot", type=float, default=0.0)
    parser.add_argument("--gallery-delta-rot-deg", "--delta-rot-deg", dest="gallery_delta_rot_deg", type=float, default=10.0)
    parser.add_argument("--gallery-n-scales", "--n-scales", dest="gallery_n_scales", type=int, default=1)
    parser.add_argument("--gallery-scale-mode", "--scale-mode", dest="gallery_scale_mode", default="linear", choices=("linear", "log"))
    parser.add_argument("--use-train-uav", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers-eval", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-rot2uniform", action="store_true")
    parser.add_argument("--query-scale2uniform", action="store_true")
    parser.add_argument("--metric", default="l2", choices=("l2", "euclidean", "cosine", "cos"))
    parser.add_argument("--bank-mode", default="discrete", choices=("discrete", "continuous_like"))
    parser.add_argument("--r-near", type=float, default=None)
    parser.add_argument("--r-far", type=float, default=None)
    parser.add_argument("--cache-gallery", action="store_true")
    parser.add_argument("--gallery-root-dir", default=None)
    parser.add_argument("--gallery-name-prefix", default=None)
    parser.add_argument("--save-per-query-csv", default="")
    parser.add_argument("--save-summary-json", default="")
    parser.add_argument("--quiet", action="store_true", help="Disable analyzer summary printing.")
    return parser


def run_stage2_response_field_analysis(
    p_yaml: str,
    stage2_ckpt: str = "",
    stage1_ckpt: str = "",
    gallery_layout_cfg=None,
    analysis_cfg=None,
    cache_gallery: bool = False,
    gallery_root_dir: Optional[str] = None,
    gallery_name_prefix: Optional[str] = None,
    parser_overrides: Optional[List[str]] = None,
):
    from trainer_depends.config.parser import get_parse
    from trainers.stage2_INGP import GridHashFitTrainer

    parser_argv = [sys.argv[0], "--p_yaml", p_yaml] + list(parser_overrides or [])
    old_argv = list(sys.argv)
    try:
        sys.argv = parser_argv
        opt = get_parse()
    finally:
        sys.argv = old_argv

    if stage1_ckpt:
        opt.load_stage1_ckpt = _resolve_cli_path(stage1_ckpt)
        if not os.path.exists(opt.load_stage1_ckpt):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {opt.load_stage1_ckpt}")

    opt.load2test = _resolve_stage2_ckpt_path(p_yaml, stage2_ckpt, opt)

    stage1_ckpt_for_test = getattr(opt, "load_stage1_ckpt", "")
    opt.load_stage1_ckpt = ""
    trainer = GridHashFitTrainer(opt=opt)
    trainer.opt.load_stage1_ckpt = stage1_ckpt_for_test
    trainer._load_checkpoints_for_test()

    layout_cfg = _normalize_gallery_layout_cfg(
        gallery_layout_cfg if gallery_layout_cfg is not None else ResponseFieldGalleryLayoutConfig()
    )
    feature_cfg = _build_stage2_feature_cfg()
    gallery_bank, stage2_layout_cfg = _build_or_load_stage2_gallery_bank(
        trainer=trainer,
        layout_cfg=layout_cfg,
        feature_cfg=feature_cfg,
        cache_gallery=bool(cache_gallery),
        gallery_root_dir=_resolve_cli_path(gallery_root_dir) if gallery_root_dir else None,
        gallery_name_prefix=gallery_name_prefix,
        ckpt_path=opt.load2test,
    )

    cfg = analysis_cfg if isinstance(analysis_cfg, ResponseFieldAnalysisConfig) else (
        ResponseFieldAnalysisConfig(**analysis_cfg) if analysis_cfg is not None else ResponseFieldAnalysisConfig()
    )
    cfg = ResponseFieldAnalysisConfig(**asdict(cfg))
    if cfg.batch_size is None:
        cfg.batch_size = int(getattr(trainer.opt, "batchsize_uav", 32))
    if cfg.num_workers is None:
        cfg.num_workers = int(getattr(trainer.opt, "num_worker_eval", 0))
    if cfg.save_per_query_csv:
        cfg.save_per_query_csv = _resolve_cli_path(cfg.save_per_query_csv)
    if cfg.save_summary_json:
        cfg.save_summary_json = _resolve_cli_path(cfg.save_summary_json)

    analyzer = ResponseFieldAnalyzer(gallery_bank=gallery_bank, trainer=trainer)
    result = analyzer.analyze(cfg)
    summary = result["summary"]
    summary["analysis_gallery_layout_cfg"] = asdict(layout_cfg)
    summary["analysis_gallery_mode"] = stage2_layout_cfg.mode
    if cfg.save_summary_json:
        analyzer._write_summary_json(cfg.save_summary_json, summary)

    result["trainer"] = trainer
    result["gallery_bank"] = gallery_bank
    result["stage2_layout_cfg"] = stage2_layout_cfg
    return result


def main(argv=None):
    args, remaining_argv = _build_arg_parser().parse_known_args(argv)

    result = run_stage2_response_field_analysis(
        p_yaml=args.p_yaml,
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        gallery_layout_cfg=ResponseFieldGalleryLayoutConfig(
            overlap=float(args.gallery_overlap),
            include_rot=bool(args.gallery_include_rot),
            include_scale=bool(args.gallery_include_scale),
            fixed_rot=float(args.gallery_fixed_rot),
            fixed_scale=args.gallery_fixed_scale,
            delta_rot_deg=float(args.gallery_delta_rot_deg),
            n_scales=int(args.gallery_n_scales),
            scale_mode=args.gallery_scale_mode,
        ),
        analysis_cfg=ResponseFieldAnalysisConfig(
            use_train_uav=bool(args.use_train_uav),
            batch_size=args.batch_size,
            num_workers=args.num_workers_eval,
            query_rot2uniform=bool(args.query_rot2uniform),
            query_scale2uniform=bool(args.query_scale2uniform),
            max_queries=args.max_queries,
            metric=args.metric,
            bank_mode=args.bank_mode,
            r_near=args.r_near,
            r_far=args.r_far,
            print_results=not bool(args.quiet),
            report_title="Response Field Analysis",
            save_per_query_csv=args.save_per_query_csv,
            save_summary_json=args.save_summary_json,
        ),
        cache_gallery=bool(args.cache_gallery),
        gallery_root_dir=args.gallery_root_dir,
        gallery_name_prefix=args.gallery_name_prefix,
        parser_overrides=remaining_argv,
    )

    summary = result["summary"]
    stage2_layout_cfg = result["stage2_layout_cfg"]
    trainer = result["trainer"]
    print("")
    print(
        "[ResponseFieldAnalysis] scene={scene} | ckpt={ckpt} | layout={mode} | metric={metric} | bank_mode={bank_mode} | N={nq}".format(
            scene=summary.get("scene_name"),
            ckpt=os.path.basename(trainer.opt.load2test),
            mode=stage2_layout_cfg.mode,
            metric=summary["metric"],
            bank_mode=summary["bank_mode"],
            nq=int(summary["n_queries"]),
        )
    )
    print(
        "[ResponseFieldAnalysis] peak_correct_rate={pcr:.4f} | mean_delta_peak={mdp:.4f} | "
        "mean_gt_rank={mgr:.4f} | mean_psr_gt={mpsr:.4f}".format(
            pcr=float(summary["peak_correct_rate"]),
            mdp=float(summary["mean_delta_peak"]),
            mgr=float(summary["mean_gt_rank"]),
            mpsr=float(summary["mean_psr_gt"]),
        )
    )
    if "mean_sampling_gap" in summary:
        print("[ResponseFieldAnalysis] mean_sampling_gap={:.4f}".format(float(summary["mean_sampling_gap"])))


if __name__ == "__main__":
    # main()
    result = run_stage2_response_field_analysis(
        p_yaml="gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1/opts.yaml",
        stage2_ckpt="gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1/epoch880_best_R@1=80.pth",
        # save_summary_json = "gen_fm_exps/analysis/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1_wRwS.json"
        gallery_layout_cfg=ResponseFieldGalleryLayoutConfig(
            overlap=0.5,
            include_rot=True,
            include_scale=False,
            #
            fixed_rot=0.0,
            fixed_scale=1.0,
            #
            delta_rot_deg= 10.0,
            n_scales= 1,
            scale_mode= "linear"
        ),
        analysis_cfg=ResponseFieldAnalysisConfig(
            query_rot2uniform=False,
            query_scale2uniform=False,
            metric="l2",
            bank_mode="continuous_like",
            batch_size=None,
            num_workers=None,
            save_summary_json="gen_fm_exps/analysis/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1_CL_wRwoS.json",
        ),
        cache_gallery=True,
    )

    summary = result["summary"]
