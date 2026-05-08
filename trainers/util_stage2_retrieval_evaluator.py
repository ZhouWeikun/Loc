from dataclasses import dataclass
import time

import torch
import torch.nn.functional as TF

from trainer_depends.utils.util_core_eval import compute_progressive_topk_acc_from_coords, print_progressive_topk_eval_results
from trainer_depends.utils.util_uav_image_transform import warp_uav_imgs


@dataclass
class Stage2RetrievalEvalConfig:
    use_train_uav: bool = False
    batch_size: int = 32
    num_workers: int = 0
    show_progress: bool = True
    query_rot2uniform: bool = False
    query_scale2uniform: bool = False
    k_values: tuple = (1, 5, 10, 20, 50)
    dist_th: float = None
    dist_lambda: float = None
    rot_th_deg: float = None
    scale_ratio_th: float = None
    max_queries: int = None
    print_results: bool = True
    report_title: str = "Stage2 Retrieval Eval"
    report_rc_meter: bool = True
    report_rot_error: bool = False
    report_scale_error: bool = False
    search_backend: str = "faiss"


class Stage2RetrievalEvaluator:
    """
    Retrieval evaluator for Stage 2 galleries.

    Responsibilities:
    - load query batches from trainer datasets
    - optionally canonicalize query rotation / scale
    - search against the given Stage 2 gallery bank
    - report rc retrieval metrics and Stage 2-specific top-1 errors
    """

    def __init__(self, trainer, gallery_bank, logger=None):
        self.trainer = trainer
        self.gallery_bank = gallery_bank
        self.device = trainer.device
        self.logger = logger or getattr(trainer, "logger", None)

    def _log(self, msg, eval_log_lines=None):
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)
        if eval_log_lines is not None:
            eval_log_lines.append(msg)

    def _enter_eval_mode(self):
        models_all = list(self.trainer.param2optimize.values()) + list(self.trainer.param2freeze.values())
        orig_modes = [m.training for m in models_all]
        for model in models_all:
            model.eval()
        return models_all, orig_modes

    def _restore_modes(self, models_all, orig_modes):
        for model, was_train in zip(models_all, orig_modes):
            model.train(was_train)

    @staticmethod
    def _normalize_search_backend(search_backend):
        search_backend = str(search_backend).strip().lower()
        valid_backends = ("faiss", "matrix")
        if search_backend not in valid_backends:
            raise ValueError(f"search_backend must be one of {valid_backends}, got {search_backend}")
        return search_backend

    def _ensure_gallery_ready(self, search_backend):
        if self.gallery_bank.coords_gallery is None:
            raise ValueError("gallery_bank.build_coords(...) must be called before retrieval evaluation.")
        if self.gallery_bank.feats_gallery is None and self.gallery_bank.faiss_index is None:
            raise ValueError("gallery_bank must have features or a FAISS index before evaluation.")
        if search_backend == "faiss" and self.gallery_bank.faiss_index is None:
            if self.gallery_bank.feats_gallery is None:
                raise ValueError("gallery_bank must have features or a FAISS index before evaluation.")
            self.gallery_bank.build_faiss_index()
        if search_backend == "matrix" and self.gallery_bank.feats_gallery is None:
            raise ValueError("matrix search requires gallery_bank.feats_gallery.")

    def _resolve_runtime_datasets(self, cfg):
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
    def _make_uav_dataloader(uav_dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            uav_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    def _prepare_query_batch(self, imgs, coords_uav, gallery_scale, cfg):
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

    def _extract_query_feats(self, imgs):
        with torch.no_grad():
            feats_q = self.trainer._get_feats_fm_imgs(imgs)
            return TF.normalize(feats_q, dim=-1)

    @staticmethod
    def _sync_if_cuda(tensor_or_device):
        device = tensor_or_device.device if torch.is_tensor(tensor_or_device) else torch.device(tensor_or_device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _search_topk(self, feats_q, coords_gallery_cpu, top_k, search_backend, gallery_feats_matrix=None):
        if search_backend == "faiss":
            _, indices = self.gallery_bank.faiss_index.search(feats_q.detach().cpu().numpy(), k=top_k)
            return coords_gallery_cpu[torch.from_numpy(indices).long()]

        q_norm_sq = torch.sum(feats_q * feats_q, dim=1, keepdim=True)
        g_norm_sq = torch.sum(gallery_feats_matrix * gallery_feats_matrix, dim=1).unsqueeze(0)
        dists_sq = q_norm_sq + g_norm_sq - 2.0 * torch.matmul(feats_q, gallery_feats_matrix.transpose(0, 1))
        _, indices = torch.topk(dists_sq, k=top_k, dim=1, largest=False, sorted=True)
        return coords_gallery_cpu[indices.detach().cpu()]

    @staticmethod
    def _resolve_scale_boundary(sat_dataset):
        if hasattr(sat_dataset, "satimgsize_scale_to_ref_m_boundary"):
            return sat_dataset.satimgsize_scale_to_ref_m_boundary
        if hasattr(sat_dataset, "satimgsize_scale_to_refm_boundary"):
            return sat_dataset.satimgsize_scale_to_refm_boundary
        return None

    def _resolve_thresholds(self, sat_dataset, cfg):
        dist_th = cfg.dist_th
        if dist_th is None:
            dist_lambda = 1.0 if cfg.dist_lambda is None else float(cfg.dist_lambda)
            dist_th = float(sat_dataset.halfimg_radius_nrc) * dist_lambda
        return {
            "norm_dist": float(dist_th),
            "rot": None if cfg.rot_th_deg is None else float(cfg.rot_th_deg),
            "scale_ratio": None if cfg.scale_ratio_th is None else float(cfg.scale_ratio_th),
        }

    @staticmethod
    def _compute_top1_arrays(coords_topk_all, coords_gt_all):
        coords_gt_expanded = coords_gt_all.unsqueeze(1)
        dist_errors = torch.norm(coords_topk_all[..., :2] - coords_gt_expanded[..., :2], p=2, dim=-1)

        rot_diff_rad = torch.abs(coords_topk_all[..., 2] - coords_gt_expanded[..., 2])
        rot_errors_rad = torch.minimum(rot_diff_rad, 2 * torch.pi - rot_diff_rad)
        rot_errors_deg = torch.rad2deg(rot_errors_rad)

        pred_scale = coords_topk_all[..., 3].clamp(min=1e-6)
        gt_scale = coords_gt_expanded[..., 3].clamp(min=1e-6)
        scale_log_errors = torch.abs(torch.log(pred_scale / gt_scale))

        return {
            "dist_top1": dist_errors[:, 0],
            "rot_deg_top1": rot_errors_deg[:, 0],
            "scale_log_top1": scale_log_errors[:, 0],
        }

    def evaluate(self, eval_cfg=None, eval_log_lines=None):
        cfg = eval_cfg if isinstance(eval_cfg, Stage2RetrievalEvalConfig) else (
            Stage2RetrievalEvalConfig(**eval_cfg) if eval_cfg is not None else Stage2RetrievalEvalConfig()
        )
        cfg.search_backend = self._normalize_search_backend(cfg.search_backend)
        self._ensure_gallery_ready(cfg.search_backend)

        scene_name, sat_dataset, uav_dataset = self._resolve_runtime_datasets(cfg)
        dataloader = self._make_uav_dataloader(
            uav_dataset,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )

        gallery_scale = float(
            self.gallery_bank.meta.get("gallery_scale_mean", getattr(sat_dataset, "satimgsize_scale_to_ref_m_mean", 1.0))
        )
        coords_gallery_cpu = self.gallery_bank.coords_gallery.cpu()
        top_k = min(max(int(k) for k in cfg.k_values), int(coords_gallery_cpu.shape[0]))
        thresholds = self._resolve_thresholds(sat_dataset, cfg)
        gallery_feats_matrix = None
        if cfg.search_backend == "matrix":
            gallery_feats_matrix = self.gallery_bank.feats_gallery.to(self.device, dtype=torch.float32)

        self._log(
            f"[Stage2RetrievalEvaluator] scene={scene_name}, "
            f"n_points={self.gallery_bank.meta.get('n_points', 0)}, "
            f"mode={self.gallery_bank.meta.get('mode', None)}, "
            f"n_rot={self.gallery_bank.meta.get('n_rot', None)}, "
            f"n_scale={self.gallery_bank.meta.get('n_scale', None)}, "
            f"search_backend={cfg.search_backend}",
            eval_log_lines=eval_log_lines,
        )

        models_all, orig_modes = self._enter_eval_mode()
        try:
            coords_topk_all = []
            coords_gt_all = []
            processed = 0
            search_time_sec_total = 0.0

            for batch in dataloader:
                if cfg.max_queries is not None and processed >= int(cfg.max_queries):
                    break

                if isinstance(batch, (list, tuple)):
                    uavimgs, coords_uav = batch[0], batch[1]
                else:
                    uavimgs, coords_uav = batch

                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)

                if cfg.max_queries is not None:
                    remain = int(cfg.max_queries) - processed
                    if uavimgs.shape[0] > remain:
                        uavimgs = uavimgs[:remain]
                        coords_uav = coords_uav[:remain]

                uavimgs, coords_uav = self._prepare_query_batch(uavimgs, coords_uav, gallery_scale, cfg)
                feats_q = self._extract_query_feats(uavimgs)
                self._sync_if_cuda(feats_q)
                search_start = time.perf_counter()
                coords_topk = self._search_topk(
                    feats_q=feats_q,
                    coords_gallery_cpu=coords_gallery_cpu,
                    top_k=top_k,
                    search_backend=cfg.search_backend,
                    gallery_feats_matrix=gallery_feats_matrix,
                )
                self._sync_if_cuda(feats_q)
                search_time_sec_total += time.perf_counter() - search_start

                coords_topk_all.append(coords_topk)
                coords_gt_all.append(coords_uav.detach().cpu())
                processed += coords_uav.shape[0]
        finally:
            self._restore_modes(models_all, orig_modes)

        if not coords_topk_all:
            raise ValueError("No valid queries were processed during evaluation.")

        coords_topk_all = torch.cat(coords_topk_all, dim=0)
        coords_gt_all = torch.cat(coords_gt_all, dim=0)
        acc_metrics_raw, shared_errors = compute_progressive_topk_acc_from_coords(
            coords_topk_all,
            coords_gt_all,
            dist_th=thresholds["norm_dist"],
            rot_th_deg=thresholds["rot"],
            scale_ratio_th=thresholds["scale_ratio"],
            k_values=cfg.k_values,
        )
        progressive_acc_metrics = (
            dict(acc_metrics_raw.get("progressive_acc_metrics", {}))
            if isinstance(acc_metrics_raw.get("progressive_acc_metrics", {}), dict)
            else {}
        )
        progressive_acc_metric_sources = (
            dict(acc_metrics_raw.get("progressive_acc_metric_sources", {}))
            if isinstance(acc_metrics_raw.get("progressive_acc_metric_sources", {}), dict)
            else {}
        )
        progressive_error_metrics = (
            dict(acc_metrics_raw.get("progressive_error_metrics", {}))
            if isinstance(acc_metrics_raw.get("progressive_error_metrics", {}), dict)
            else {}
        )
        progressive_error_metric_sources = (
            dict(acc_metrics_raw.get("progressive_error_metric_sources", {}))
            if isinstance(acc_metrics_raw.get("progressive_error_metric_sources", {}), dict)
            else {}
        )
        legacy_acc_metrics_source = str(acc_metrics_raw.get("legacy_acc_metrics_source", "dist_rot_scale_recall"))
        metrics = {
            str(key): float(value)
            for key, value in acc_metrics_raw.items()
            if str(key).startswith("top") and str(key).endswith("_acc")
        }
        report_meta = {
            "integrate_scale": thresholds["scale_ratio"] is not None,
            "scale_select_mode": None,
            "legacy_acc_metrics_source": legacy_acc_metrics_source,
            "progressive_acc_metric_sources": progressive_acc_metric_sources,
            "progressive_error_metric_sources": progressive_error_metric_sources,
            "progressive_recall_policy": {
                "dist_recall": "dist<=dist_th",
                "dist_rot_recall": "dist<=dist_th and rot<=rot_th",
                "dist_rot_scale_recall": "dist<=dist_th and rot<=rot_th and scale<=scale_ratio_th",
                "rot_fallback_to_dist": thresholds["rot"] is None,
                "scale_fallback_to_dist_rot": thresholds["scale_ratio"] is None,
            },
        }

        top1_arrays = self._compute_top1_arrays(coords_topk_all, coords_gt_all)
        dist_top1 = top1_arrays["dist_top1"]
        rot_deg_top1 = top1_arrays["rot_deg_top1"]
        scale_log_top1 = top1_arrays["scale_log_top1"]

        dist_meter_top1 = (
            float(sat_dataset.halfimg_radius_meter) * dist_top1 / float(sat_dataset.halfimg_radius_nrc)
        )
        scale_boundary = self._resolve_scale_boundary(sat_dataset)
        if scale_boundary is not None:
            scale_min = max(float(scale_boundary[0]), 1e-6)
            scale_max = max(float(scale_boundary[1]), scale_min)
            denom = torch.log(torch.tensor(scale_max / scale_min, dtype=torch.float32))
            if float(denom.item()) > 0:
                scale_normed_top1 = scale_log_top1 / denom
            else:
                scale_normed_top1 = torch.zeros_like(scale_log_top1)
        else:
            scale_normed_top1 = scale_log_top1

        if cfg.print_results:
            report_title = cfg.report_title
            if scene_name is not None:
                report_title = f"{report_title} [{scene_name}]"
            print_progressive_topk_eval_results(
                {
                    **metrics,
                    "progressive_acc_metrics": progressive_acc_metrics,
                    "legacy_acc_metrics_source": legacy_acc_metrics_source,
                    "progressive_acc_metric_sources": progressive_acc_metric_sources,
                    "progressive_error_metrics": progressive_error_metrics,
                    "progressive_error_metric_sources": progressive_error_metric_sources,
                },
                shared_errors,
                thresholds,
                report_title=report_title,
                report_meta=report_meta,
                log_lines=eval_log_lines,
            )
            self._log(
                f"RC Error Top-1: mean={dist_top1.mean().item():.5f}, median={dist_top1.median().item():.5f}",
                eval_log_lines=eval_log_lines,
            )
            if cfg.report_rc_meter:
                self._log(
                    f"RC Error Top-1 (meter): mean={dist_meter_top1.mean().item():.2f}m, "
                    f"median={dist_meter_top1.median().item():.2f}m",
                    eval_log_lines=eval_log_lines,
                )
            if cfg.report_rot_error:
                self._log(
                    f"Rotation Error Top-1: mean={rot_deg_top1.mean().item():.2f}deg, "
                    f"median={rot_deg_top1.median().item():.2f}deg",
                    eval_log_lines=eval_log_lines,
                )
            if cfg.report_scale_error:
                self._log(
                    f"Scale Error Top-1 (normalized log): mean={scale_normed_top1.mean().item():.5f}, "
                    f"median={scale_normed_top1.median().item():.5f}",
                    eval_log_lines=eval_log_lines,
                )
            avg_search_sec_per_query = search_time_sec_total / max(int(coords_gt_all.shape[0]), 1)
            self._log(
                f"Retrieval Search Time ({cfg.search_backend}): total={search_time_sec_total:.6f}s, "
                f"avg={avg_search_sec_per_query * 1000.0:.6f}ms/query, "
                f"top_k={top_k}, gallery_size={int(coords_gallery_cpu.shape[0])}",
                eval_log_lines=eval_log_lines,
            )

        recall_at_k = {
            int(k): float(metrics.get(f"top{int(k)}_acc", 0.0)) / 100.0
            for k in cfg.k_values
        }
        avg_search_sec_per_query = search_time_sec_total / max(int(coords_gt_all.shape[0]), 1)
        return {
            "scene_name": scene_name,
            "n_queries": int(coords_gt_all.shape[0]),
            "n_eval": int(coords_gt_all.shape[0]),
            "report_title": str(cfg.report_title),
            "k_values": tuple(int(k) for k in cfg.k_values),
            "thresholds": thresholds,
            "metrics": metrics,
            "shared_errors": shared_errors,
            "report_meta": report_meta,
            "progressive_acc_metrics": progressive_acc_metrics,
            "progressive_acc_metric_sources": progressive_acc_metric_sources,
            "progressive_error_metrics": progressive_error_metrics,
            "progressive_error_metric_sources": progressive_error_metric_sources,
            "legacy_acc_metrics_source": legacy_acc_metrics_source,
            "coords_topk": coords_topk_all,
            "coords_gt": coords_gt_all,
            "recall@k": recall_at_k,
            "error_rc_norm": float(dist_top1.mean().item()),
            "error_rc_norm_median": float(dist_top1.median().item()),
            "error_rc_meter": float(dist_meter_top1.mean().item()),
            "error_rc_meter_median": float(dist_meter_top1.median().item()),
            "error_rot_deg": float(rot_deg_top1.mean().item()),
            "error_rot_deg_median": float(rot_deg_top1.median().item()),
            "error_scale_ratio": float(torch.exp(scale_log_top1).mean().item()),
            "error_scale_ratio_median": float(torch.exp(scale_log_top1).median().item()),
            "error_scale_normed": float(scale_normed_top1.mean().item()),
            "error_scale_normed_median": float(scale_normed_top1.median().item()),
            "search_backend": str(cfg.search_backend),
            "retrieval_search_time_sec_total": float(search_time_sec_total),
            "retrieval_search_num_queries": int(coords_gt_all.shape[0]),
            "retrieval_search_avg_sec_per_query": float(avg_search_sec_per_query),
            "retrieval_search_avg_ms_per_query": float(avg_search_sec_per_query * 1000.0),
            "retrieval_search_top_k": int(top_k),
            "retrieval_search_gallery_size": int(coords_gallery_cpu.shape[0]),
            "runtime_gallery_summary": self.gallery_bank.summary(),
        }
