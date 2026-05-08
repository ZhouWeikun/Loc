from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import Subset

from trainer_depends.utils.util_core_eval import (
    compute_progressive_topk_acc_from_coords,
    print_progressive_topk_eval_results,
)
from trainers.util_stage1_gallery_manager import Stage1ReferenceGalleryDownsampleConfig
from trainer_depends.utils.util_uav_image_transform import warp_uav_imgs


@dataclass
class Stage1RetrievalEvalConfig:
    # Whether to evaluate on the training UAV set instead of the test split.
    use_train_uav: bool = False
    # Query dataloader batch size during retrieval evaluation.
    batch_size: int = 32
    # Number of dataloader workers used for query loading.
    num_workers: int = 0
    # Whether to rotate each query to a canonical orientation before retrieval.
    query_rot2uniform: bool = False
    # Whether to rescale each query to the gallery reference scale before retrieval.
    query_scale2uniform: bool = False
    # Top-k cutoffs used when reporting retrieval accuracy.
    k_values: tuple = (1, 5, 10, 20, 50, 100)
    # Distance threshold on nr/nc coordinates; None means derive it from the dataset.
    dist_th: float = None
    # Rotation threshold in degrees; None means rotation is not used in hit judgement.
    rot_th_deg: float = None
    # Multiplicative scale-ratio threshold; None means scale is not used in hit judgement.
    scale_ratio_th: float = None
    # Optional cap on the number of evaluated queries.
    max_queries: int = None
    # Optional runtime gallery downsample config; the original gallery bank remains unchanged.
    gallery_downsample_cfg: object = None
    # Optional second-stage subset on the already selected query split.
    # Example for wingtra interval82 correction:
    #   first use the existing 8:2 split as test set, then split that test set by
    #   interval 50:50 and keep the second half as the final evaluation queries.
    query_subset_mode: str = None
    query_subset_train_ratio: float = None
    query_subset_take: str = "test"
    query_subset_random_seed: int = 2026
    # Whether to print a formatted report after evaluation.
    print_results: bool = True
    # Title shown in the printed evaluation report.
    report_title: str = "Stage1 Retrieval Eval"


class Stage1RetrievalEvaluator:
    """
    A clean retrieval evaluator that consumes an existing Stage1ReferenceGalleryBank.

    Responsibilities:
    - load query batches from the trainer datasets
    - extract query features
    - search against the given gallery bank
    - evaluate top-k coordinate retrieval metrics
    """

    def __init__(self, trainer, gallery_bank, logger=None):
        self.trainer = trainer
        self.gallery_bank = gallery_bank
        self.device = trainer.device
        self.logger = logger or getattr(trainer, "logger", None)

    def _log(self, msg, eval_log_lines=None):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
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

    def _ensure_gallery_ready(self):
        if self.gallery_bank.coords_gallery is None:
            raise ValueError("gallery_bank.build_coords(...) must be called before retrieval evaluation.")
        if self.gallery_bank.faiss_index is None:
            if self.gallery_bank.feats_gallery is None:
                raise ValueError("gallery_bank must have features or a FAISS index before evaluation.")
            self.gallery_bank.build_faiss_index()

    def _build_runtime_gallery_bank(self, cfg):
        runtime_bank = self.gallery_bank
        if cfg.gallery_downsample_cfg is not None:
            runtime_bank = self.gallery_bank.downsample(cfg.gallery_downsample_cfg)
        if runtime_bank.faiss_index is None:
            if runtime_bank.feats_gallery is None:
                raise ValueError("Runtime gallery bank must have features or a FAISS index.")
            runtime_bank.build_faiss_index()
        return runtime_bank

    def _make_uav_dataloader(self, uav_dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            uav_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    @staticmethod
    def _compute_split_indices(n_samples, train_ratio=0.9, split_mode='segment', random_seed=2026):
        n_samples = int(n_samples)
        train_ratio = float(train_ratio)
        split_mode = str(split_mode).strip().lower()

        if n_samples <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if split_mode not in ('segment', 'interval', 'random'):
            raise ValueError(f"split_mode must be 'segment', 'interval', or 'random', got {split_mode!r}")

        indices = np.arange(n_samples, dtype=np.int64)
        if split_mode == 'segment':
            n_train = int(n_samples * train_ratio)
            n_train = min(max(n_train, 1), max(n_samples - 1, 1))
            return indices[:n_train], indices[n_train:]

        if split_mode == 'random':
            n_train = int(n_samples * train_ratio)
            n_train = min(max(n_train, 1), max(n_samples - 1, 1))
            rng = np.random.RandomState(int(random_seed))
            perm = rng.permutation(indices)
            train_indices = np.sort(perm[:n_train])
            test_indices = np.sort(perm[n_train:])
            return train_indices, test_indices

        ratio_frac = Fraction(str(train_ratio)).limit_denominator(1000)
        period = int(ratio_frac.denominator)
        train_per_period = int(ratio_frac.numerator)
        offset_in_period = indices % period
        train_mask = offset_in_period < train_per_period
        train_indices = indices[train_mask]
        test_indices = indices[~train_mask]
        if len(train_indices) == 0 or len(test_indices) == 0:
            n_train = int(n_samples * train_ratio)
            n_train = min(max(n_train, 1), max(n_samples - 1, 1))
            return indices[:n_train], indices[n_train:]
        return train_indices, test_indices

    def _maybe_subset_query_dataset(self, uav_dataset, cfg, use_train_uav):
        if cfg.query_subset_mode in (None, "", "none") or cfg.query_subset_train_ratio is None:
            return uav_dataset, {
                "enabled": False,
                "source_split": "train" if use_train_uav else "test",
                "n_before": int(len(uav_dataset)),
                "n_after": int(len(uav_dataset)),
            }

        train_indices, test_indices = self._compute_split_indices(
            n_samples=len(uav_dataset),
            train_ratio=cfg.query_subset_train_ratio,
            split_mode=cfg.query_subset_mode,
            random_seed=cfg.query_subset_random_seed,
        )
        take = str(cfg.query_subset_take).strip().lower()
        if take == "train":
            selected_rel = train_indices
        elif take == "test":
            selected_rel = test_indices
        else:
            raise ValueError(f"query_subset_take must be 'train' or 'test', got {cfg.query_subset_take!r}")

        subset_dataset = Subset(uav_dataset, selected_rel.tolist())
        subset_meta = {
            "enabled": True,
            "source_split": "train" if use_train_uav else "test",
            "mode": str(cfg.query_subset_mode),
            "train_ratio": float(cfg.query_subset_train_ratio),
            "take": take,
            "random_seed": int(cfg.query_subset_random_seed),
            "n_before": int(len(uav_dataset)),
            "n_after": int(len(subset_dataset)),
            "selected_rel_indices_head": selected_rel[:10].tolist(),
        }
        base_attr = "train_indices" if use_train_uav else "test_indices"
        if hasattr(uav_dataset, base_attr):
            base_indices = np.asarray(getattr(uav_dataset, base_attr), dtype=np.int64)
            subset_meta["selected_source_indices_head"] = base_indices[selected_rel[:10]].tolist()
        return subset_dataset, subset_meta

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

    def _resolve_thresholds(self, sat_dataset, cfg):
        dist_th = cfg.dist_th
        if dist_th is None:
            dist_th = float(sat_dataset.halfimg_radius_nrc) * 1.1
        nrc2meter = None
        dist_th_meter = None
        if hasattr(sat_dataset, "halfimg_radius_meter") and hasattr(sat_dataset, "halfimg_radius_nrc"):
            nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
            dist_th_meter = float(dist_th) * nrc2meter
        return {
            "norm_dist": float(dist_th),
            "dist_meter": None if dist_th_meter is None else float(dist_th_meter),
            "nrc2meter": None if nrc2meter is None else float(nrc2meter),
            "rot": None if cfg.rot_th_deg is None else float(cfg.rot_th_deg),
            "scale_ratio": None if cfg.scale_ratio_th is None else float(cfg.scale_ratio_th),
        }

    def evaluate_scene(self, scene_name=None, eval_cfg=None, eval_log_lines=None):
        self._ensure_gallery_ready()

        cfg = eval_cfg if isinstance(eval_cfg, Stage1RetrievalEvalConfig) else (
            Stage1RetrievalEvalConfig(**eval_cfg) if eval_cfg is not None else Stage1RetrievalEvalConfig()
        )
        if cfg.gallery_downsample_cfg is not None and not isinstance(cfg.gallery_downsample_cfg, Stage1ReferenceGalleryDownsampleConfig):
            cfg.gallery_downsample_cfg = Stage1ReferenceGalleryDownsampleConfig(**cfg.gallery_downsample_cfg)

        if not hasattr(self.trainer, "sat_datasets"):
            self.trainer._init_datasets(create_train_loader=False)

        if scene_name is None:
            scene_name = self.gallery_bank.meta.get("scene_name", None)
        if scene_name is None:
            raise ValueError("scene_name must be provided when gallery metadata does not include it.")
        if scene_name not in self.trainer.sat_datasets:
            raise KeyError(f"Unknown scene_name: {scene_name}")

        sat_dataset = self.trainer.sat_datasets[scene_name]
        uav_dataset = self.trainer.uav_datasets_train[scene_name] if cfg.use_train_uav else self.trainer.uav_datasets_test[scene_name]
        uav_dataset, query_subset_meta = self._maybe_subset_query_dataset(
            uav_dataset=uav_dataset,
            cfg=cfg,
            use_train_uav=cfg.use_train_uav,
        )
        dataloader = self._make_uav_dataloader(uav_dataset, batch_size=int(cfg.batch_size), num_workers=int(cfg.num_workers))

        runtime_gallery_bank = self._build_runtime_gallery_bank(cfg)
        gallery_scale = float(runtime_gallery_bank.meta.get("gallery_scale_mean", sat_dataset.satimgsize_scale_to_ref_m_mean))
        coords_gallery_cpu = runtime_gallery_bank.coords_gallery.cpu()
        top_k = min(max(int(k) for k in cfg.k_values), int(coords_gallery_cpu.shape[0]))
        thresholds = self._resolve_thresholds(sat_dataset, cfg)

        self._log(
            f"[Stage1RetrievalEvaluator] scene={scene_name}, "
            f"source_n_points={self.gallery_bank.meta.get('n_points', 0)}, "
            f"runtime_n_points={runtime_gallery_bank.meta.get('n_points', 0)}, "
            f"source_n_bins_4d={self.gallery_bank.meta.get('n_bins_4d', None)}, "
            f"runtime_n_bins_4d={runtime_gallery_bank.meta.get('n_bins_4d', None)}, "
            f"downsample_cfg={None if cfg.gallery_downsample_cfg is None else cfg.gallery_downsample_cfg}"
            ,
            eval_log_lines=eval_log_lines,
        )

        models_all, orig_modes = self._enter_eval_mode()
        try:
            coords_topk_all = []
            coords_gt_all = []
            feats_q_all = []
            processed = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    uavimgs, coords_uav = batch[0], batch[1]
                else:
                    uavimgs, coords_uav = batch

                if cfg.max_queries is not None and processed >= int(cfg.max_queries):
                    break

                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)
                if cfg.max_queries is not None:
                    remain = int(cfg.max_queries) - processed
                    if uavimgs.shape[0] > remain:
                        uavimgs = uavimgs[:remain]
                        coords_uav = coords_uav[:remain]

                uavimgs, coords_uav = self._prepare_query_batch(uavimgs, coords_uav, gallery_scale, cfg)
                feats_q = self._extract_query_feats(uavimgs)
                _, indices = runtime_gallery_bank.faiss_index.search(feats_q.detach().cpu().numpy(), k=top_k)
                coords_topk = coords_gallery_cpu[torch.from_numpy(indices).long()]

                coords_topk_all.append(coords_topk)
                coords_gt_all.append(coords_uav.detach().cpu())
                feats_q_all.append(feats_q.detach().cpu())
                processed += coords_uav.shape[0]
        finally:
            self._restore_modes(models_all, orig_modes)

        if not coords_topk_all:
            raise ValueError("No valid queries were processed during evaluation.")

        coords_topk_all = torch.cat(coords_topk_all, dim=0)
        coords_gt_all = torch.cat(coords_gt_all, dim=0)
        feats_q_all = torch.cat(feats_q_all, dim=0)
        acc_metrics_raw, err_stats = compute_progressive_topk_acc_from_coords(
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
        legacy_acc_metrics_source = str(acc_metrics_raw.get("legacy_acc_metrics_source", "dist_rot_scale_recall"))
        acc_metrics = {
            str(key): float(value)
            for key, value in acc_metrics_raw.items()
            if str(key).startswith("top") and str(key).endswith("_acc")
        }
        report_meta = {
            "integrate_scale": thresholds["scale_ratio"] is not None,
            "scale_select_mode": None,
            "legacy_acc_metrics_source": legacy_acc_metrics_source,
            "progressive_acc_metric_sources": progressive_acc_metric_sources,
            "query_subset": query_subset_meta,
            "progressive_recall_policy": {
                "dist_recall": "dist<=dist_th",
                "dist_rot_recall": "dist<=dist_th and rot<=rot_th",
                "dist_rot_scale_recall": "dist<=dist_th and rot<=rot_th and scale<=scale_ratio_th",
                "rot_fallback_to_dist": thresholds["rot"] is None,
                "scale_fallback_to_dist_rot": thresholds["scale_ratio"] is None,
            },
        }

        if cfg.print_results:
            report_title = f"{cfg.report_title} [{scene_name}]"
            print_progressive_topk_eval_results(
                {
                    **acc_metrics,
                    "progressive_acc_metrics": progressive_acc_metrics,
                    "legacy_acc_metrics_source": legacy_acc_metrics_source,
                    "progressive_acc_metric_sources": progressive_acc_metric_sources,
                },
                err_stats,
                thresholds,
                report_title=report_title,
                report_meta=report_meta,
                log_lines=eval_log_lines,
            )

        return {
            "scene_name": scene_name,
            "n_queries": int(coords_gt_all.shape[0]),
            "k_values": tuple(int(k) for k in cfg.k_values),
            "thresholds": thresholds,
            "report_title": f"{cfg.report_title} [{scene_name}]",
            "report_meta": report_meta,
            "metrics": acc_metrics,
            "errors": err_stats,
            "acc_metrics": acc_metrics,
            "progressive_acc_metrics": progressive_acc_metrics,
            "err_stats": err_stats,
            "n_eval": int(coords_gt_all.shape[0]),
            "coords_topk": coords_topk_all,
            "coords_gt": coords_gt_all,
            "feats_query": feats_q_all,
            "runtime_gallery_summary": runtime_gallery_bank.summary(),
        }
