from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as TF

from trainers.util_core_eval import compute_topk_acc_from_coords, print_topk_eval_results
from trainers.util_stage1_gallery_manager import Stage1ReferenceGalleryDownsampleConfig
from trainers.util_stage1_others import warp_uav_imgs


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
        return {
            "norm_dist": float(dist_th),
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
                processed += coords_uav.shape[0]
        finally:
            self._restore_modes(models_all, orig_modes)

        if not coords_topk_all:
            raise ValueError("No valid queries were processed during evaluation.")

        coords_topk_all = torch.cat(coords_topk_all, dim=0)
        coords_gt_all = torch.cat(coords_gt_all, dim=0)
        metrics, errors = compute_topk_acc_from_coords(
            coords_topk_all,
            coords_gt_all,
            dist_th=thresholds["norm_dist"],
            rot_th_deg=thresholds["rot"],
            scale_ratio_th=thresholds["scale_ratio"],
            k_values=cfg.k_values,
        )

        if cfg.print_results:
            report_title = f"{cfg.report_title} [{scene_name}]"
            print_topk_eval_results(
                metrics,
                errors,
                thresholds,
                report_title=report_title,
                log_lines=eval_log_lines,
            )

        return {
            "scene_name": scene_name,
            "n_queries": int(coords_gt_all.shape[0]),
            "k_values": tuple(int(k) for k in cfg.k_values),
            "thresholds": thresholds,
            "metrics": metrics,
            "errors": errors,
            "coords_topk": coords_topk_all,
            "coords_gt": coords_gt_all,
            "runtime_gallery_summary": runtime_gallery_bank.summary(),
        }
