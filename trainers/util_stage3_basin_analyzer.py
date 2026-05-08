import json
import os
import time

import numpy as np
import torch
import tqdm

from trainers.util_core_eval import _compute_coord_error_tensors
from trainers.util_stage3_loc_manager import Stage3FineLocManager
from trainers.util_stage3_multi_start_CMAES_by_evotorch import MultiStartCMAESEvoTorchRefiner


class Stage3BasinAnalyzer:
    """Estimate Stage-3 CMA-ES basin size from one population per query."""

    def __init__(
        self,
        trainer,
        eval_thresh_cfg=None,
        sample_radius_rc=(5.0, 5.0),
        sample_radius_rot_deg=0.0,
        sample_radius_scale_ratio=0.0,
        fix_rot=True,
        fix_scale=True,
        seed=None,
        chunk_size=2048,
        temperature=0.5,
    ):
        self.trainer = trainer
        self.device = trainer.device
        self.eval_thresh_cfg = None if eval_thresh_cfg is None else dict(eval_thresh_cfg)
        self.manager = Stage3FineLocManager(
            trainer,
            eval_thresh_cfg=self.eval_thresh_cfg,
            chunk_size=chunk_size,
            temperature=temperature,
        )
        self.refiner = MultiStartCMAESEvoTorchRefiner(coords_processor=trainer.coord_normer, device=self.device)
        self.sample_radius_rc = self._parse_pair(sample_radius_rc, "sample_radius_rc")
        self.sample_radius_rot_deg = float(sample_radius_rot_deg)
        self.sample_radius_scale_ratio = float(sample_radius_scale_ratio)
        self.fix_rot = bool(fix_rot)
        self.fix_scale = bool(fix_scale)
        self.seed = None if seed is None or str(seed).lower() in {"", "none", "null"} else int(seed)
        self.generator = torch.Generator(device=self.device)
        if self.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(self.seed)

    @staticmethod
    def _parse_pair(value, name):
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
            value = [float(p) for p in parts]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a pair, got {value!r}")
        return (float(value[0]), float(value[1]))

    @staticmethod
    def _to_jsonable(value):
        if torch.is_tensor(value):
            return Stage3BasinAnalyzer._to_jsonable(value.detach().cpu().numpy())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): Stage3BasinAnalyzer._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Stage3BasinAnalyzer._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _format_optional_float(value, fmt=".6f"):
        if value is None:
            return "None"
        return format(float(value), fmt)

    def _resolve_thresholds(self):
        cfg = dict(self.manager.final_eval_cfg)
        dist_th = cfg.get("dist_th", None)
        if dist_th is None:
            dist_lambda = float(cfg.get("dist_lambda", 1.1))
            dist_th = float(self.trainer.sat_dataset.halfimg_radius_nrc) * dist_lambda
        nrc2meter = None
        sat_dataset = getattr(self.trainer, "sat_dataset", None)
        if (
            sat_dataset is not None
            and hasattr(sat_dataset, "halfimg_radius_meter")
            and hasattr(sat_dataset, "halfimg_radius_nrc")
        ):
            nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
        return {
            "dist_th": float(dist_th),
            "dist_th_meter": None if nrc2meter is None else float(dist_th) * nrc2meter,
            "nrc2meter": nrc2meter,
            "rot_th": None if cfg.get("rot_th", None) is None else float(cfg["rot_th"]),
            "scale_ratio_th": None if cfg.get("scale_ratio_th", None) is None else float(cfg["scale_ratio_th"]),
        }

    def _iter_query_batches(self, n_samples, use_train_uav, query_ids=None, batch_size=16, shuffle=False):
        dataset = self.trainer.uav_dataset_train if use_train_uav else self.trainer.uav_dataset_test
        if query_ids is not None:
            ids = [int(x) for x in query_ids]
        else:
            if n_samples is None:
                n_samples = len(dataset)
            else:
                n_samples = min(int(n_samples), len(dataset))
            ids = list(range(int(n_samples)))
            if shuffle:
                ids = torch.randperm(int(n_samples)).tolist()

        for start in range(0, len(ids), int(batch_size)):
            batch_ids = ids[start:start + int(batch_size)]
            imgs = []
            coords = []
            for idx in batch_ids:
                img, coord = dataset[int(idx)]
                imgs.append(img)
                coords.append(coord)
            yield batch_ids, torch.stack(imgs, dim=0), torch.stack(coords, dim=0)

    def _query_metadata(self, query_index, use_train_uav):
        dataset = self.trainer.uav_dataset_train if use_train_uav else self.trainer.uav_dataset_test
        split_name = "train" if use_train_uav else "test"
        idx = int(query_index)
        meta = {
            "query_index": idx,
            "split": split_name,
            "dataset_name": str(getattr(dataset, "name", "")),
            "p_uavinfo_json": str(getattr(dataset, "p_uavinfo_json", "")),
            "p_uav_geocsv": str(getattr(dataset, "p_uav_geocsv", "")),
        }

        paths_attr = "uavimg_paths_train" if use_train_uav else "uavimg_paths_test"
        paths = getattr(dataset, paths_attr, None)
        if paths is not None and 0 <= idx < len(paths):
            path = str(paths[idx])
            meta["query_path"] = path
            meta["query_filename"] = os.path.basename(path)

        split_indices_attr = "train_indices" if use_train_uav else "test_indices"
        split_indices = getattr(dataset, split_indices_attr, None)
        if split_indices is not None and 0 <= idx < len(split_indices):
            meta["source_index"] = int(split_indices[idx])

        row_indices_attr = "uav_row_indices_train" if use_train_uav else "uav_row_indices_test"
        row_indices = getattr(dataset, row_indices_attr, None)
        if row_indices is not None and 0 <= idx < len(row_indices):
            meta["csv_row_index"] = int(row_indices[idx])

        df_attr = "uav_df_train" if use_train_uav else "uav_df_test"
        df = getattr(dataset, df_attr, None)
        if df is not None and 0 <= idx < len(df):
            row = df.iloc[idx]
            for key in ("filename", "latitude", "longitude", "rotdeg_fm_north_anticlock", "h_cover_m"):
                if key in row:
                    meta[key] = self._to_jsonable(row[key])

        latlons_attr = "uav_lonlats_train" if use_train_uav else "uav_lonlats_test"
        latlons = getattr(dataset, latlons_attr, None)
        if latlons is not None and 0 <= idx < len(latlons):
            meta["latlon"] = self._to_jsonable(latlons[idx])

        return meta

    def _score_pair_fn(self, cma_prob_mode):
        cma_prob_mode = self.manager.validate_prob_mode("cma_prob_mode", cma_prob_mode)

        def _fn(query_feats_mxc, coords_raw_mx4):
            with torch.no_grad():
                scores = self.manager.score_candidates(
                    query_feats=query_feats_mxc,
                    coords_batch=coords_raw_mx4.unsqueeze(1),
                    mode=cma_prob_mode,
                    normalize=False,
                )
            return torch.nan_to_num(scores.reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)

        return _fn

    def _success_mask(self, coords_pred, coords_gt, thresholds):
        dist_errors, rot_errors_deg, scale_ratio = _compute_coord_error_tensors(coords_pred, coords_gt)
        success = dist_errors <= float(thresholds["dist_th"])
        if thresholds["rot_th"] is not None:
            success = success & (rot_errors_deg <= float(thresholds["rot_th"]))
        if thresholds["scale_ratio_th"] is not None:
            success = success & (scale_ratio <= float(thresholds["scale_ratio_th"]))
        return success, dist_errors, rot_errors_deg, scale_ratio

    def _project_raw_coords(self, coords):
        coords_proj = coords.clone()
        sat_dataset = getattr(self.trainer, "sat_dataset", None)
        if sat_dataset is not None:
            if hasattr(sat_dataset, "nr2sample_range"):
                nr_min, nr_max = sat_dataset.nr2sample_range
                coords_proj[..., 0] = coords_proj[..., 0].clamp(float(nr_min), float(nr_max))
            if hasattr(sat_dataset, "nc2sample_range"):
                nc_min, nc_max = sat_dataset.nc2sample_range
                coords_proj[..., 1] = coords_proj[..., 1].clamp(float(nc_min), float(nc_max))
            scale_bounds = getattr(sat_dataset, "satimgsize_scale_to_ref_m_boundary", None)
            if scale_bounds is None:
                scale_bounds = getattr(sat_dataset, "satimgsize_scale_to_refm_boundary", None)
            if scale_bounds is not None:
                coords_proj[..., 3] = coords_proj[..., 3].clamp(float(scale_bounds[0]), float(scale_bounds[1]))
            else:
                coords_proj[..., 3] = coords_proj[..., 3].clamp(min=1e-6)
        else:
            coords_proj[..., 3] = coords_proj[..., 3].clamp(min=1e-6)
        coords_proj[..., 2] = torch.remainder(coords_proj[..., 2], 2 * torch.pi)
        return coords_proj

    def _stage1_5_radius_raw(self, center_coords, radius_scale=1.0):
        radius = torch.zeros((4,), device=self.device, dtype=torch.float32)
        radius[0] = float(self.sample_radius_rc[0])
        radius[1] = float(self.sample_radius_rc[1])
        if not self.fix_rot:
            radius[2] = float(np.deg2rad(self.sample_radius_rot_deg))
        if not self.fix_scale:
            scale_ref = center_coords[..., 3].detach().to(self.device, dtype=torch.float32).reshape(-1)
            scale_base = float(torch.median(scale_ref).item()) if scale_ref.numel() > 0 else 1.0
            radius[3] = max(0.0, float(self.sample_radius_scale_ratio)) * max(scale_base, 1e-6)
        return radius * float(radius_scale)

    def _sample_stage1_5_population(self, centers, radius_raw, num_particles):
        batch_size = int(centers.shape[0])
        num_particles = int(num_particles)
        noise = torch.rand((batch_size, num_particles, 4), generator=self.generator, device=self.device) * 2.0 - 1.0
        radius = radius_raw.reshape(1, 1, 4)
        coords = centers[:, None, :] + noise * radius
        if self.fix_rot:
            coords[..., 2] = centers[:, None, 2]
        if self.fix_scale:
            coords[..., 3] = centers[:, None, 3]
        return self._project_raw_coords(coords)

    @staticmethod
    def _weighted_cyclic_rot(rot_values, weights):
        sin_v = torch.sum(torch.sin(rot_values) * weights, dim=1)
        cos_v = torch.sum(torch.cos(rot_values) * weights, dim=1)
        return torch.remainder(torch.atan2(sin_v, cos_v), 2 * torch.pi)

    def _elite_center(self, coords, scores, elite_ratio):
        batch_size, num_particles, _ = coords.shape
        elite_count = max(1, int(np.ceil(num_particles * float(elite_ratio))))
        elite_count = min(elite_count, num_particles)
        elite_scores, elite_idx = torch.topk(scores, k=elite_count, dim=1, largest=True)
        elite_coords = torch.gather(coords, dim=1, index=elite_idx.unsqueeze(-1).expand(-1, -1, 4))
        weights = torch.ones((batch_size, elite_count), device=self.device, dtype=torch.float32) / max(elite_count, 1)
        center = torch.sum(elite_coords * weights.unsqueeze(-1), dim=1)
        center[:, 2] = self._weighted_cyclic_rot(elite_coords[..., 2], weights)
        best_idx = torch.argmax(scores, dim=1)
        best_coords = torch.gather(coords, dim=1, index=best_idx.reshape(batch_size, 1, 1).expand(-1, -1, 4)).squeeze(1)
        best_scores = torch.gather(scores, dim=1, index=best_idx.reshape(batch_size, 1)).squeeze(1)
        return self._project_raw_coords(center), best_coords, best_scores

    def _run_stage1_5_mode_refine_batch(
        self,
        query_feats,
        center_coords,
        score_pair_fn,
        num_particles,
        n_iterations,
        elite_ratio,
        radius_decay,
        move_stand,
    ):
        centers = center_coords.squeeze(1).clone()
        history_best_coords = centers.clone()
        history_best_scores = torch.full((centers.shape[0],), -1e9, device=self.device, dtype=torch.float32)
        final_population_coords = None
        final_population_scores = None
        move_stand = str(move_stand).strip().lower()
        if move_stand not in {"best", "elite_sum"}:
            raise ValueError(f"stage1_5 move_stand must be 'best' or 'elite_sum', got {move_stand}")

        n_iterations = max(1, int(n_iterations))
        for round_idx in range(n_iterations):
            radius_raw = self._stage1_5_radius_raw(centers, radius_scale=float(radius_decay) ** int(round_idx))
            coords = self._sample_stage1_5_population(centers, radius_raw, num_particles=num_particles)
            flat_query_feats = query_feats[:, None, :].expand(-1, int(num_particles), -1).reshape(-1, query_feats.shape[-1])
            flat_scores = score_pair_fn(flat_query_feats, coords.reshape(-1, 4))
            scores = flat_scores.reshape(centers.shape[0], int(num_particles))
            elite_centers, round_best_coords, round_best_scores = self._elite_center(coords, scores, elite_ratio=elite_ratio)

            improve = round_best_scores > history_best_scores
            history_best_scores = torch.where(improve, round_best_scores, history_best_scores)
            history_best_coords = torch.where(improve.reshape(-1, 1), round_best_coords, history_best_coords)
            centers = history_best_coords.clone() if move_stand == "best" else elite_centers
            if self.fix_rot:
                centers[:, 2] = center_coords[:, 0, 2]
            if self.fix_scale:
                centers[:, 3] = center_coords[:, 0, 3]

            final_population_coords = coords
            final_population_scores = scores

        return {
            "best_coords": history_best_coords.unsqueeze(1),
            "best_scores": history_best_scores.unsqueeze(1),
            "final_population_coords": final_population_coords.unsqueeze(1),
            "final_population_scores": final_population_scores.unsqueeze(1),
        }

    def _optimize_dims(self):
        dims = [0, 1]
        if not self.fix_rot:
            dims.append(2)
        if not self.fix_scale:
            dims.append(3)
        return tuple(dims)

    def run(
        self,
        n_samples=256,
        use_train_uav=False,
        query_ids=None,
        num_particles=128,
        query_batch_size=8,
        shuffle=False,
        cma_sigma0=0.2,
        cma_popsize=16,
        cma_iters=20,
        cma_variant="Sep-CMA",
        cma_prob_mode="product",
        cma_enable_early_stop=True,
        cma_early_stop_patience=5,
        optimizer_backend="cma_es",
        stage1_5_iters=None,
        stage1_5_elite_ratio=0.125,
        stage1_5_radius_decay=1.0,
        stage1_5_move_stand="elite_sum",
        save_particles=False,
        output_dir=None,
        progress=True,
    ):
        thresholds = self._resolve_thresholds()
        optimize_dims = self._optimize_dims()
        score_pair_fn = self._score_pair_fn(cma_prob_mode)
        optimizer_backend = str(optimizer_backend).strip().lower()
        if optimizer_backend in {"cma", "cma-es", "cmaes"}:
            optimizer_backend = "cma_es"
        if optimizer_backend in {"stage15", "stage1.5", "mode_refine", "stage1_5_mode_refine"}:
            optimizer_backend = "stage1_5"
        if optimizer_backend not in {"cma_es", "stage1_5"}:
            raise ValueError(f"optimizer_backend must be 'cma_es' or 'stage1_5', got {optimizer_backend}")
        stage1_5_iters = int(cma_iters if stage1_5_iters is None else stage1_5_iters)

        per_query = []
        particle_records = []
        all_success_counts = []
        all_particle_counts = []
        total_queries = 0
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        batch_iter = self._iter_query_batches(
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            query_ids=query_ids,
            batch_size=query_batch_size,
            shuffle=shuffle,
        )
        pbar = None
        if progress:
            total = None
            if query_ids is not None:
                total = len(query_ids)
            elif n_samples is not None:
                total = int(n_samples)
            pbar = tqdm.tqdm(
                total=total,
                desc="Stage3 basin",
                unit="query",
                dynamic_ncols=True,
                leave=True,
            )

        try:
            for batch_ids, imgs, coords_gt in batch_iter:
                imgs = imgs.to(self.device)
                coords_gt = coords_gt.to(self.device, dtype=torch.float32)
                query_feats = self.trainer._get_feats_fm_imgs(imgs)
                center_coords = coords_gt[:, None, :].clone()

                with torch.no_grad():
                    center_scores = score_pair_fn(query_feats, center_coords.reshape(-1, 4)).reshape(-1, 1)

                if optimizer_backend == "cma_es":
                    cma_seed_batch = None if self.seed is None else int(self.seed) + int(total_queries) * 100000
                    final = self.refiner.refine_batch_queries(
                        query_feats=query_feats,
                        seed_coords_raw_batch=center_coords,
                        score_pair_fn=score_pair_fn,
                        sigma0=cma_sigma0,
                        popsize=int(num_particles),
                        n_iterations=cma_iters,
                        maximize=True,
                        cma_seed=cma_seed_batch,
                        cma_variant=cma_variant,
                        enable_early_stop=cma_enable_early_stop,
                        early_stop_patience=cma_early_stop_patience,
                        return_diagnostics=True,
                        optimize_dims=optimize_dims,
                    )
                else:
                    final = self._run_stage1_5_mode_refine_batch(
                        query_feats=query_feats,
                        center_coords=center_coords,
                        score_pair_fn=score_pair_fn,
                        num_particles=num_particles,
                        n_iterations=stage1_5_iters,
                        elite_ratio=stage1_5_elite_ratio,
                        radius_decay=stage1_5_radius_decay,
                        move_stand=stage1_5_move_stand,
                    )
                best_coords = final["best_coords"]
                best_scores = final["best_scores"]
                final_population_coords = final["final_population_coords"].squeeze(1)
                final_population_scores = final["final_population_scores"].squeeze(1)

                center_success, center_dist, center_rot, center_scale = self._success_mask(center_coords, coords_gt, thresholds)
                final_success, final_dist, final_rot, final_scale = self._success_mask(
                    final_population_coords,
                    coords_gt,
                    thresholds,
                )
                best_success, best_dist, best_rot, best_scale = self._success_mask(best_coords, coords_gt, thresholds)

                for local_idx, query_index in enumerate(batch_ids):
                    success_count = int(final_success[local_idx].sum().item())
                    center_success_count = int(center_success[local_idx].sum().item())
                    particle_count = int(final_success.shape[1])
                    all_success_counts.append(success_count)
                    all_particle_counts.append(particle_count)
                    total_queries += 1
                    query_meta = self._query_metadata(query_index, use_train_uav=use_train_uav)
                    per_query.append({
                        **query_meta,
                        "query_index": int(query_index),
                        "gt_coord": self._to_jsonable(coords_gt[local_idx]),
                        "num_particles": particle_count,
                        "center_success": bool(center_success_count),
                        "center_score": float(center_scores[local_idx, 0].item()),
                        "population_success_count": success_count,
                        "population_success_rate": float(success_count) / max(particle_count, 1),
                        "final_success_count": success_count,
                        "final_success_rate": float(success_count) / max(particle_count, 1),
                        "center_dist": float(center_dist[local_idx, 0].item()),
                        "center_rot_deg": float(center_rot[local_idx, 0].item()),
                        "center_scale_ratio": float(center_scale[local_idx, 0].item()),
                        "mean_final_dist": float(final_dist[local_idx].mean().item()),
                        "median_final_dist": float(torch.median(final_dist[local_idx]).item()),
                        "mean_final_rot_deg": float(final_rot[local_idx].mean().item()),
                        "median_final_rot_deg": float(torch.median(final_rot[local_idx]).item()),
                        "mean_final_scale_ratio": float(final_scale[local_idx].mean().item()),
                        "median_final_scale_ratio": float(torch.median(final_scale[local_idx]).item()),
                        "best_coord": self._to_jsonable(best_coords[local_idx, 0]),
                        "best_score": float(best_scores[local_idx, 0].item()),
                        "best_success": bool(best_success[local_idx, 0].item()),
                        "best_dist": float(best_dist[local_idx, 0].item()),
                        "best_rot_deg": float(best_rot[local_idx, 0].item()),
                        "best_scale_ratio": float(best_scale[local_idx, 0].item()),
                    })
                    if save_particles:
                        for particle_idx in range(particle_count):
                            particle_records.append({
                                "query_index": int(query_index),
                                "particle_index": int(particle_idx),
                                "final_coord": self._to_jsonable(final_population_coords[local_idx, particle_idx]),
                                "final_score": float(final_population_scores[local_idx, particle_idx].item()),
                                "final_success": bool(final_success[local_idx, particle_idx].item()),
                                "final_dist": float(final_dist[local_idx, particle_idx].item()),
                                "final_rot_deg": float(final_rot[local_idx, particle_idx].item()),
                                "final_scale_ratio": float(final_scale[local_idx, particle_idx].item()),
                            })

                if pbar is not None:
                    processed_queries = len(batch_ids)
                    total_particles_done = int(sum(all_particle_counts))
                    total_success_done = int(sum(all_success_counts))
                    rate = 0.0 if total_particles_done <= 0 else total_success_done / float(total_particles_done)
                    pbar.set_postfix({
                        "pop": int(num_particles),
                        "hit": total_success_done,
                        "rate": f"{rate * 100.0:.2f}%",
                    })
                    pbar.update(processed_queries)
        finally:
            if pbar is not None:
                pbar.close()

        total_particles = int(sum(all_particle_counts))
        total_success = int(sum(all_success_counts))
        query_rates = [row["final_success_rate"] for row in per_query]
        successful_queries = sum(1 for row in per_query if int(row["final_success_count"]) > 0)
        result = {
            "summary": {
                "started_at": started_at,
                "n_queries": int(total_queries),
                "num_particles": int(num_particles),
                "total_particles": total_particles,
                "total_success": total_success,
                "overall_success_rate": 0.0 if total_particles <= 0 else float(total_success) / float(total_particles),
                "successful_queries": int(successful_queries),
                "query_localization_success_rate": (
                    0.0 if total_queries <= 0 else float(successful_queries) / float(total_queries)
                ),
                "mean_query_success_rate": 0.0 if not query_rates else float(np.mean(query_rates)),
                "median_query_success_rate": 0.0 if not query_rates else float(np.median(query_rates)),
                "min_query_success_rate": 0.0 if not query_rates else float(np.min(query_rates)),
                "max_query_success_rate": 0.0 if not query_rates else float(np.max(query_rates)),
            },
            "config": {
                "eval_thresh_cfg": self.eval_thresh_cfg,
                "thresholds_resolved": thresholds,
                "sample_radius_rc": list(self.sample_radius_rc),
                "sample_radius_rot_deg": self.sample_radius_rot_deg,
                "sample_radius_scale_ratio": self.sample_radius_scale_ratio,
                "fix_rot": self.fix_rot,
                "fix_scale": self.fix_scale,
                "seed": self.seed,
                "optimizer_backend": optimizer_backend,
                "cma_sigma0": cma_sigma0,
                "cma_popsize": int(num_particles),
                "legacy_cma_popsize_arg": int(cma_popsize),
                "cma_iters": int(cma_iters),
                "cma_variant": str(cma_variant),
                "cma_prob_mode": str(cma_prob_mode),
                "stage1_5_iters": int(stage1_5_iters),
                "stage1_5_elite_ratio": float(stage1_5_elite_ratio),
                "stage1_5_radius_decay": float(stage1_5_radius_decay),
                "stage1_5_move_stand": str(stage1_5_move_stand),
                "optimize_dims": list(optimize_dims),
            },
            "per_query": per_query,
        }
        if save_particles:
            result["particles"] = particle_records

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "stage3_basin_analysis.json")
            report_path = os.path.join(output_dir, "stage3_basin_analysis.txt")
            result["summary"]["save_path"] = save_path
            result["summary"]["report_path"] = report_path
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self._to_jsonable(result), f, indent=2, ensure_ascii=False)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.format_text_report(result))

        return result

    def format_text_report(self, result):
        summary = result["summary"]
        config = result["config"]
        thresholds = config["thresholds_resolved"]
        lines = [
            "Stage3 Basin Analysis",
            "=" * 80,
            f"started_at: {summary['started_at']}",
            "",
            "[Config]",
            f"num_particles/popsize: {config['cma_popsize']}",
            f"optimizer_backend: {config.get('optimizer_backend', 'cma_es')}",
            f"cma_iters: {config['cma_iters']}",
            f"cma_sigma0: {config['cma_sigma0']}",
            f"cma_variant: {config['cma_variant']}",
            f"cma_prob_mode: {config['cma_prob_mode']}",
            f"stage1_5_iters: {config.get('stage1_5_iters')}",
            f"stage1_5_elite_ratio: {config.get('stage1_5_elite_ratio')}",
            f"stage1_5_radius_decay: {config.get('stage1_5_radius_decay')}",
            f"stage1_5_move_stand: {config.get('stage1_5_move_stand')}",
            f"optimize_dims: {config['optimize_dims']}",
            f"fix_rot: {config['fix_rot']}",
            f"fix_scale: {config['fix_scale']}",
            f"seed: {config['seed']}",
            f"sample_radius_rc: {config['sample_radius_rc']}",
            f"sample_radius_rot_deg: {config['sample_radius_rot_deg']}",
            f"sample_radius_scale_ratio: {config['sample_radius_scale_ratio']}",
            f"eval_thresh_cfg: {config['eval_thresh_cfg']}",
            "",
            "[Resolved Thresholds]",
            f"dist_th_nrc: {self._format_optional_float(thresholds.get('dist_th'))}",
            f"dist_th_meter: {self._format_optional_float(thresholds.get('dist_th_meter'))}",
            f"rot_th_deg: {self._format_optional_float(thresholds.get('rot_th'))}",
            f"scale_ratio_th: {self._format_optional_float(thresholds.get('scale_ratio_th'))}",
            "",
            "[Summary]",
            f"n_queries: {summary['n_queries']}",
            f"total_particles: {summary['total_particles']}",
            f"total_success_particles: {summary['total_success']}",
            f"particle_success_rate: {summary['overall_success_rate'] * 100.0:.3f}%",
            f"successful_queries: {summary['successful_queries']}",
            f"query_localization_success_rate: {summary['query_localization_success_rate'] * 100.0:.3f}%",
            f"mean_query_particle_success_rate: {summary['mean_query_success_rate'] * 100.0:.3f}%",
            f"median_query_particle_success_rate: {summary['median_query_success_rate'] * 100.0:.3f}%",
            f"min_query_particle_success_rate: {summary['min_query_success_rate'] * 100.0:.3f}%",
            f"max_query_particle_success_rate: {summary['max_query_success_rate'] * 100.0:.3f}%",
            "",
            "[Per Query]",
            "query_index\tsource_index\tcsv_row_index\tfilename\tsuccess_particles\ttotal_particles\tparticle_success_rate\tquery_success\tbest_success\tbest_dist\tbest_rot_deg\tbest_scale_ratio\tpath",
        ]
        for row in result["per_query"]:
            query_success = int(row["final_success_count"]) > 0
            lines.append(
                f"{row['query_index']}\t"
                f"{row.get('source_index', '')}\t"
                f"{row.get('csv_row_index', '')}\t"
                f"{row.get('query_filename', row.get('filename', ''))}\t"
                f"{row['final_success_count']}\t"
                f"{row['num_particles']}\t"
                f"{row['final_success_rate'] * 100.0:.3f}%\t"
                f"{query_success}\t"
                f"{row['best_success']}\t"
                f"{row['best_dist']:.6f}\t"
                f"{row['best_rot_deg']:.6f}\t"
                f"{row['best_scale_ratio']:.6f}\t"
                f"{row.get('query_path', '')}"
            )
        lines.append("")
        return "\n".join(lines)
