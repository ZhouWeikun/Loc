import numpy as np
import torch
import tqdm

from trainer_depends.utils.util_core_eval import compute_topk_acc_from_coords, print_topk_eval_results


def compute_topN_acc_given_threshold(
    coords_pred,
    coords_gt,
    dist_th,
    rot_th_deg=None,
    scale_ratio_th=None,
    k_values=None,
):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    dist_metrics, errors = compute_topk_acc_from_coords(
        coords_pred,
        coords_gt,
        dist_th=dist_th,
        rot_th_deg=None,
        scale_ratio_th=None,
        k_values=k_values,
    )
    progressive_acc_metrics = {"dist_recall": {str(k): float(v) for k, v in dist_metrics.items()}}
    progressive_acc_metric_sources = {"dist_recall": "computed"}

    if rot_th_deg is None:
        progressive_acc_metrics["dist_rot_recall"] = dict(progressive_acc_metrics["dist_recall"])
        progressive_acc_metric_sources["dist_rot_recall"] = "alias_of_dist_recall"
    else:
        dist_rot_metrics, _ = compute_topk_acc_from_coords(
            coords_pred,
            coords_gt,
            dist_th=dist_th,
            rot_th_deg=rot_th_deg,
            scale_ratio_th=None,
            k_values=k_values,
        )
        progressive_acc_metrics["dist_rot_recall"] = {str(k): float(v) for k, v in dist_rot_metrics.items()}
        progressive_acc_metric_sources["dist_rot_recall"] = "computed"

    if scale_ratio_th is None:
        progressive_acc_metrics["dist_rot_scale_recall"] = dict(progressive_acc_metrics["dist_rot_recall"])
        progressive_acc_metric_sources["dist_rot_scale_recall"] = "alias_of_dist_rot_recall"
    else:
        dist_rot_scale_metrics, _ = compute_topk_acc_from_coords(
            coords_pred,
            coords_gt,
            dist_th=dist_th,
            rot_th_deg=rot_th_deg,
            scale_ratio_th=scale_ratio_th,
            k_values=k_values,
        )
        progressive_acc_metrics["dist_rot_scale_recall"] = {
            str(k): float(v) for k, v in dist_rot_scale_metrics.items()
        }
        progressive_acc_metric_sources["dist_rot_scale_recall"] = "computed"

    if scale_ratio_th is not None:
        legacy_acc_metrics_source = "dist_rot_scale_recall"
    elif rot_th_deg is not None:
        legacy_acc_metrics_source = "dist_rot_recall"
    else:
        legacy_acc_metrics_source = "dist_recall"

    legacy_metrics = dict(progressive_acc_metrics[legacy_acc_metrics_source])
    legacy_metrics["progressive_acc_metrics"] = progressive_acc_metrics
    legacy_metrics["legacy_acc_metrics_source"] = legacy_acc_metrics_source
    legacy_metrics["progressive_acc_metric_sources"] = progressive_acc_metric_sources
    return legacy_metrics, errors


def print_topN_acc_results(metrics, errors, thresholds, report_meta=None, report_title="Fine Accuracy Report"):
    report_meta = dict(report_meta or {})
    progressive_acc_metrics = metrics.get("progressive_acc_metrics", None) if isinstance(metrics, dict) else None
    if not isinstance(progressive_acc_metrics, dict):
        print_topk_eval_results(metrics, errors, thresholds, report_title=report_title, report_meta=report_meta)
        return

    sections = [
        ("dist_recall", "Dist Recall", {"norm_dist": thresholds.get("norm_dist"), "rot": None, "scale_ratio": None}),
        ("dist_rot_recall", "Dist+Rot Recall", {"norm_dist": thresholds.get("norm_dist"), "rot": thresholds.get("rot"), "scale_ratio": None}),
        ("dist_rot_scale_recall", "Dist+Rot+Scale Recall", thresholds),
    ]
    for group_key, section_title, section_thresholds in sections:
        section_metrics = progressive_acc_metrics.get(group_key, None)
        if isinstance(section_metrics, dict):
            section_meta = dict(report_meta)
            section_meta["integrate_scale"] = section_thresholds.get("scale_ratio") is not None
            print_topk_eval_results(
                section_metrics,
                errors,
                section_thresholds,
                report_title=f"{report_title} | {section_title}",
                report_meta=section_meta,
            )


def compute_top_k_accuracy(pred_pdf, gt_labels, k_values=None, dim_order="HWO"):
    if k_values is None:
        k_values = [1, 4, 9, 16, 50]
    if isinstance(pred_pdf, torch.Tensor):
        pred_pdf = pred_pdf.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    n_samples = pred_pdf.shape[0]
    if pred_pdf.ndim == 4:
        target_order = "HWO"
        if dim_order != target_order:
            idx_map = {char: i + 1 for i, char in enumerate(dim_order)}
            pred_pdf = np.transpose(pred_pdf, [0] + [idx_map[c] for c in target_order])
        pred_pdf_flat = pred_pdf.reshape(n_samples, -1)
    elif pred_pdf.ndim == 2:
        pred_pdf_flat = pred_pdf
    else:
        raise ValueError(f"pred_pdf must be [N,D1,D2,D3] or [N,C], got shape {pred_pdf.shape}")

    gt_labels = np.asarray(gt_labels).reshape(-1).astype(np.int64)[:n_samples]
    pred_pdf_flat = pred_pdf_flat[: len(gt_labels)]
    order = np.argsort(-pred_pdf_flat, axis=1)
    results = {}
    for k in k_values:
        k = min(int(k), pred_pdf_flat.shape[1])
        hits = (order[:, :k] == gt_labels[:, None]).any(axis=1)
        results[f"top{k}_acc"] = float(hits.mean() * 100.0) if len(hits) else 0.0
    return results


def print_accuracy_results(results, title="3D定位准确率"):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    for key, value in results.items():
        print(f"{key}: {float(value):.2f}")
    print(f"{'=' * 60}\n")


class Stage3FineLocManager:
    """
    Single entry-point manager for Stage 3 fine localization.

    It keeps the original "one class does three things" workflow:
    1. build coarse candidates
    2. score candidates with projector / ingp style metrics
    3. evaluate and report retrieval / localization results
    """

    def __init__(self, trainer, eval_thresh_cfg=None, chunk_size=2048, temperature=0.5):
        self.trainer = trainer
        self.temperature = temperature
        self.chunk_size = chunk_size

        self.eval_cfg = {"dist_lambda": 1.0, "dist_th": None, "rot_th": 10.0, "scale_ratio_th": None}
        self.final_eval_cfg = {"dist_lambda": 1.1, "dist_th": None, "rot_th": 11.0, "scale_ratio_th": None}
        if eval_thresh_cfg is not None:
            if not isinstance(eval_thresh_cfg, dict):
                raise ValueError("eval_thresh_cfg must be a dict or None.")
            if "dist_th" not in eval_thresh_cfg and "dist_th_meter" in eval_thresh_cfg:
                if not hasattr(self.trainer.sat_dataset, "halfimg_radius_meter") or not hasattr(self.trainer.sat_dataset, "halfimg_radius_nrc"):
                    raise ValueError("dist_th_meter requires sat_dataset.halfimg_radius_meter and sat_dataset.halfimg_radius_nrc")
                eval_thresh_cfg = dict(eval_thresh_cfg)
                meter2nrc = float(self.trainer.sat_dataset.halfimg_radius_nrc) / max(
                    float(self.trainer.sat_dataset.halfimg_radius_meter), 1e-8
                )
                eval_thresh_cfg["dist_th"] = float(eval_thresh_cfg["dist_th_meter"]) * meter2nrc
            if "scale_ratio_th" not in eval_thresh_cfg and "scale_th" in eval_thresh_cfg:
                eval_thresh_cfg = dict(eval_thresh_cfg)
                old_scale_th = eval_thresh_cfg.pop("scale_th")
                eval_thresh_cfg["scale_ratio_th"] = None if old_scale_th is None else (1.0 + float(old_scale_th))
            for key in ("dist_lambda", "dist_th", "rot_th", "scale_ratio_th"):
                if key in eval_thresh_cfg:
                    self.eval_cfg[key] = eval_thresh_cfg[key]
                    self.final_eval_cfg[key] = eval_thresh_cfg[key]
            if self.final_eval_cfg["dist_th"] is not None:
                dist_lambda = float(self.final_eval_cfg["dist_th"]) / max(
                    float(self.trainer.sat_dataset.halfimg_radius_nrc), 1e-8
                )
                self.eval_cfg["dist_lambda"] = dist_lambda
                self.final_eval_cfg["dist_lambda"] = dist_lambda

    # ============================================================
    # Candidate Building
    # ============================================================
    def build_candidates(self, n_bins_4d=None, n_bins_scale_mode="linear"):
        if n_bins_4d is None:
            coords_candidates, _ = self.trainer.subspace_sampler.sample_all_subspaces_gpu(
                n_points_per_subspace=1,
                use_fine=False,
                rand_offset=False,
            )
            coords_candidates_flat = coords_candidates.view(-1, 4)
            n_coarse = self.trainer.subspace_sampler.n_coarse
        else:
            coords_candidates_flat = self.trainer.subspace_sampler.sample_uniform_grid_by_bins(
                n_bins_4d,
                device=self.trainer.device,
                scale_mode=n_bins_scale_mode,
            )
            coords_candidates = coords_candidates_flat.view(1, -1, 4)
            n_coarse = np.array(n_bins_4d, dtype=np.int32)

        n_coarse_3d = n_coarse[:3]
        coords_reshaped = coords_candidates.squeeze(0).reshape(*n_coarse, 4)
        cell_centers_3d = coords_reshaped[:, :, :, 0, :3]
        return {
            "coords_candidates_flat": coords_candidates_flat,
            "n_coarse": n_coarse,
            "n_coarse_3d": n_coarse_3d,
            "cell_centers_3d": cell_centers_3d,
        }

    def make_test_loader(self, use_train_uav=False, n_samples=256, shuffle=False):
        dataset = self.trainer.uav_dataset_train if use_train_uav else self.trainer.uav_dataset_test
        if n_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(int(n_samples), len(dataset))

        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
        return dataset, n_samples, test_loader

    def coords_to_gt_flat_idx(self, coords_gt, n_coarse, use_custom_grid=False):
        n_nr3, n_nc3, n_rot3 = [int(x) for x in n_coarse[:3]]
        if not use_custom_grid:
            gt_indices_flat = self.trainer.subspace_sampler.coords_to_coarse_indices(coords_gt)
            gt_indices_multi = self.trainer.subspace_sampler.coarse_indices_to_multi(gt_indices_flat)
            gt_nr = gt_indices_multi[:, 0]
            gt_nc = gt_indices_multi[:, 1]
            gt_rot = gt_indices_multi[:, 2]
        else:
            gt_bin_indices = self.trainer.subspace_sampler._coords_to_bin_indices(
                coords_gt.cpu().numpy(),
                n_coarse,
            )
            gt_bin_indices = torch.from_numpy(gt_bin_indices).to(self.trainer.device)
            gt_nr = gt_bin_indices[:, 0]
            gt_nc = gt_bin_indices[:, 1]
            gt_rot = gt_bin_indices[:, 2]
        return gt_nr * (n_nc3 * n_rot3) + gt_nc * n_rot3 + gt_rot

    # ============================================================
    # Candidate Scoring
    # ============================================================
    @staticmethod
    def validate_prob_mode(mode_name, mode_value):
        mode_value = str(mode_value).lower()
        if mode_value not in ("ingp", "projector", "product"):
            raise ValueError(
                f"{mode_name} must be one of ('ingp','projector','product'), got {mode_value}"
            )
        return mode_value

    def score_candidates(self, query_feats, coords_batch, mode="ingp", normalize=False, projector_metric="dist"):
        mode = self.validate_prob_mode("mode", mode)
        projector_metric = str(projector_metric).lower()
        if projector_metric not in ("dist", "possibility"):
            raise ValueError(f"Unknown projector_metric: {projector_metric}")

        if mode in ("ingp", "product"):
            dist_ingp = self.trainer._compute_metric_from_ingp(
                query_feats,
                coords_batch,
                coord_space="raw",
                chunk_size=self.chunk_size,
                metric="dist",
            )
            score_ingp = (2 - dist_ingp).clamp(min=0) / 2
        else:
            score_ingp = None

        if mode in ("projector", "product"):
            if mode == "projector" and projector_metric == "possibility":
                score_proj = self.trainer._compute_metric_from_query_and_points(
                    query_feats,
                    coords_batch,
                    metric="possibility",
                    coord_space="raw",
                    temperature=self.temperature,
                    chunk_size=self.chunk_size,
                    feat_type="projector",
                )
            else:
                dist_proj = self.trainer._compute_metric_from_query_and_points(
                    query_feats,
                    coords_batch,
                    metric="dist",
                    coord_space="raw",
                    temperature=self.temperature,
                    chunk_size=self.chunk_size,
                    feat_type="projector",
                )
                score_proj = (2 - dist_proj).clamp(min=0) / 2
        else:
            score_proj = None

        if mode == "ingp":
            score = score_ingp
        elif mode == "projector":
            score = score_proj
        else:
            score = score_ingp.clamp(min=0.1) * score_proj

        if normalize:
            score = score / (score.sum(dim=-1, keepdim=True) + 1e-8)
        return score

    # ============================================================
    # Coarse Retrieval Runtime
    # ============================================================
    def run_l0_coarse_retrieval(
        self,
        n_samples=256,
        use_train_uav=False,
        shuffle=False,
        n_bins_4d=None,
        n_bins_scale_mode="linear",
        l0_prob_mode="ingp",
        save_pred_pdf=False,
        save_tag="prefilter",
        report_title="Spatial Classification Results (wo eval scale)",
        show_progress=False,
        progress_desc="L0 coarse retrieval",
    ):
        _, n_samples, test_loader = self.make_test_loader(
            use_train_uav=use_train_uav,
            n_samples=n_samples,
            shuffle=shuffle,
        )
        use_custom_grid = n_bins_4d is not None
        candidate_info = self.build_candidates(n_bins_4d=n_bins_4d, n_bins_scale_mode=n_bins_scale_mode)
        coords_candidates_flat = candidate_info["coords_candidates_flat"]
        n_coarse = candidate_info["n_coarse"]
        n_coarse_3d = candidate_info["n_coarse_3d"]
        cell_centers_3d = candidate_info["cell_centers_3d"]

        processed = 0
        pred_pdf_3d_list = []
        q_label_3d_list = []
        coords_gt_list = []
        feats_vis_list = []
        l0_prob_4d_list = []

        batch_iter = test_loader
        batch_progress = None
        if bool(show_progress):
            loader_batch_size = test_loader.batch_size or 1
            total_batches = (int(n_samples) + loader_batch_size - 1) // loader_batch_size
            batch_progress = tqdm.tqdm(
                test_loader,
                total=total_batches,
                desc=str(progress_desc),
                leave=True,
            )
            batch_iter = batch_progress

        with torch.no_grad():
            for batch in batch_iter:
                if processed >= n_samples:
                    break

                imgs = batch[0].to(self.trainer.device)
                coords_gt = batch[1].to(self.trainer.device)
                batch_size = imgs.shape[0]
                remain = n_samples - processed
                if batch_size > remain:
                    imgs = imgs[:remain]
                    coords_gt = coords_gt[:remain]
                    batch_size = remain

                feats_vis = self.trainer._get_feats_fm_imgs(imgs)
                possibilities = self.score_candidates(
                    query_feats=feats_vis,
                    coords_batch=coords_candidates_flat,
                    mode=l0_prob_mode,
                    normalize=True,
                    projector_metric="possibility",
                )

                possibilities_reshaped = possibilities.reshape(batch_size, *n_coarse)
                logits_3d = possibilities_reshaped.sum(dim=-1)
                logits_3d = logits_3d / (logits_3d.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
                prob_flat = logits_3d.view(batch_size, -1)

                gt_flat_idx = self.coords_to_gt_flat_idx(
                    coords_gt,
                    n_coarse,
                    use_custom_grid=use_custom_grid,
                )

                pred_pdf_3d_list.append(prob_flat.cpu())
                q_label_3d_list.append(gt_flat_idx.cpu())
                coords_gt_list.append(coords_gt.cpu())
                feats_vis_list.append(feats_vis.cpu())
                l0_prob_4d_list.append(possibilities.cpu())
                processed += batch_size

        if batch_progress is not None:
            batch_progress.close()

        pred_pdf_3d_all = torch.cat(pred_pdf_3d_list, dim=0)
        q_label_3d_all = torch.cat(q_label_3d_list, dim=0)
        coords_gt_all = torch.cat(coords_gt_list, dim=0).to(self.trainer.device)
        feats_vis_all = torch.cat(feats_vis_list, dim=0).to(self.trainer.device)
        l0_prob_4d_all = torch.cat(l0_prob_4d_list, dim=0).to(self.trainer.device)

        self.maybe_save_pred_pdf(
            save_pred_pdf=save_pred_pdf,
            pred_pdf_3d_all=pred_pdf_3d_all,
            q_label_3d_all=q_label_3d_all,
            coords_gt_all=coords_gt_all,
            n_coarse_3d=n_coarse_3d,
            cell_centers_3d=cell_centers_3d,
            use_train_uav=use_train_uav,
            tag=save_tag,
            temperature=self.temperature,
        )
        spatial_classification_metrics = self.report_spatial_classification(
            pred_pdf_3d_all=pred_pdf_3d_all,
            q_label_3d_all=q_label_3d_all,
            title=report_title,
        )

        return {
            "pred_pdf_3d_all": pred_pdf_3d_all,
            "q_label_3d_all": q_label_3d_all,
            "coords_gt_all": coords_gt_all,
            "feats_vis_all": feats_vis_all,
            "l0_prob_4d_all": l0_prob_4d_all,
            "coords_candidates_flat": coords_candidates_flat,
            "n_coarse": n_coarse,
            "n_coarse_3d": n_coarse_3d,
            "cell_centers_3d": cell_centers_3d,
            "spatial_classification_report": {
                "report_title": str(report_title),
                "metrics": spatial_classification_metrics,
            },
        }

    # ============================================================
    # Evaluation / Reporting
    # ============================================================
    def evaluate_and_report(
        self,
        coords_pred,
        coords_gt_source,
        tag="Eval",
        dist_lambda=1.0,
        rot_th=10.0,
        scale_ratio_th=None,
        scale_select_mode=None,
        return_details=False,
    ):
        if isinstance(coords_gt_source, list):
            if len(coords_gt_source) > 0:
                coords_gt_all = torch.cat(coords_gt_source, dim=0).to(coords_pred.device)
            else:
                print(f"[{tag}] Warning: GT list is empty!")
                return None
        else:
            coords_gt_all = coords_gt_source.to(coords_pred.device)

        dist_th_nrc = self.final_eval_cfg.get("dist_th", None)
        if dist_th_nrc is None:
            dist_th_nrc = self.trainer.sat_dataset.halfimg_radius_nrc * dist_lambda
        thresh_cfg = {
            "norm_dist": float(dist_th_nrc),
            "dist_meter": None,
            "nrc2meter": None,
            "rot": rot_th,
            "scale_ratio": scale_ratio_th,
        }
        if hasattr(self.trainer.sat_dataset, "halfimg_radius_meter") and hasattr(self.trainer.sat_dataset, "halfimg_radius_nrc"):
            nrc2meter = float(self.trainer.sat_dataset.halfimg_radius_meter) / max(
                float(self.trainer.sat_dataset.halfimg_radius_nrc), 1e-8
            )
            thresh_cfg["nrc2meter"] = float(nrc2meter)
            thresh_cfg["dist_meter"] = float(thresh_cfg["norm_dist"]) * nrc2meter

        base_k_values = [1, 5, 10, 16, 32, 64, 128, 256, 512]
        if coords_pred.ndim == 2:
            k_max = 1
        else:
            k_max = int(coords_pred.shape[1])
        target_k_values = [k for k in base_k_values if k <= k_max]
        if not target_k_values:
            target_k_values = [1]

        min_len = min(len(coords_pred), len(coords_gt_all))
        if min_len == 0:
            print("No data to evaluate.")
            return None

        coords_pred = coords_pred[:min_len]
        coords_gt_all = coords_gt_all[:min_len]

        acc_metrics_raw, err_stats = compute_topN_acc_given_threshold(
            coords_pred=coords_pred,
            coords_gt=coords_gt_all,
            dist_th=thresh_cfg["norm_dist"],
            rot_th_deg=thresh_cfg["rot"],
            scale_ratio_th=thresh_cfg["scale_ratio"],
            k_values=target_k_values,
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
            "integrate_scale": scale_ratio_th is not None,
            "scale_select_mode": scale_select_mode,
            "legacy_acc_metrics_source": legacy_acc_metrics_source,
            "progressive_acc_metric_sources": progressive_acc_metric_sources,
            "progressive_recall_policy": {
                "dist_recall": "dist<=dist_th",
                "dist_rot_recall": "dist<=dist_th and rot<=rot_th",
                "dist_rot_scale_recall": (
                    "dist<=dist_th and rot<=rot_th and scale<=scale_ratio_th"
                ),
                "rot_fallback_to_dist": thresh_cfg["rot"] is None,
                "scale_fallback_to_dist_rot": thresh_cfg["scale_ratio"] is None,
            },
        }
        print_topN_acc_results(
            {
                **acc_metrics,
                "progressive_acc_metrics": progressive_acc_metrics,
                "legacy_acc_metrics_source": legacy_acc_metrics_source,
                "progressive_acc_metric_sources": progressive_acc_metric_sources,
            },
            err_stats,
            thresh_cfg,
            report_title=str(tag),
            report_meta=report_meta,
        )
        if return_details:
            return {
                "report_title": str(tag),
                "thresholds": dict(thresh_cfg),
                "report_meta": dict(report_meta),
                "acc_metrics": acc_metrics,
                "progressive_acc_metrics": progressive_acc_metrics,
                "err_stats": err_stats,
                "n_eval": int(min_len),
            }
        return acc_metrics

    def report_spatial_classification(self, pred_pdf_3d_all, q_label_3d_all, title):
        single_frame_results = compute_top_k_accuracy(
            pred_pdf_3d_all.cpu().numpy(),
            q_label_3d_all,
            k_values=[1, 8, 27, 64, 128, 256, 512, 1024],
            dim_order="HWO",
        )
        print_accuracy_results(single_frame_results, title=title)
        return single_frame_results

    def maybe_save_pred_pdf(
        self,
        save_pred_pdf,
        pred_pdf_3d_all,
        q_label_3d_all,
        coords_gt_all,
        n_coarse_3d,
        cell_centers_3d,
        use_train_uav,
        tag,
        temperature,
    ):
        if not save_pred_pdf:
            return
        data_type = "train" if use_train_uav else "test"
        self.trainer._save_pred_pdf_3d(
            pred_pdf_3d_all=pred_pdf_3d_all,
            q_label_3d_all=q_label_3d_all,
            coords_gt_all=coords_gt_all,
            n_coarse_3d=n_coarse_3d,
            cell_centers_3d=cell_centers_3d,
            temperature=temperature,
            data_type=data_type,
            tag=tag,
        )

    def evaluate_pred_pdf_topk(self, pred_pdf_3d_all, cell_centers_3d, n_coarse_3d, coords_gt_all):
        coords_topk = None
        scores_topk = None
        pred_pdf_cpu = pred_pdf_3d_all.detach().cpu()
        if pred_pdf_cpu.shape[1] == 0:
            print("[Eval_pred_pdf] pred_pdf_3d_all has zero columns, skip.")
            return coords_topk, scores_topk

        k_max = min(512, pred_pdf_cpu.shape[1])
        topk_res = torch.topk(pred_pdf_cpu, k=k_max, dim=1, largest=True)
        topk_idx = topk_res.indices
        scores_topk = topk_res.values.to(self.trainer.device)

        cell_centers_cpu = (
            cell_centers_3d.detach().cpu()
            if isinstance(cell_centers_3d, torch.Tensor)
            else torch.from_numpy(cell_centers_3d)
        )
        n_nr, n_nc, n_rot = [int(x) for x in n_coarse_3d]
        idx_rot = topk_idx % n_rot
        idx_nc = (topk_idx // n_rot) % n_nc
        idx_nr = topk_idx // (n_nc * n_rot)
        coords_topk_xyz = cell_centers_cpu[idx_nr, idx_nc, idx_rot]

        coords_gt_cpu = coords_gt_all.detach().cpu()
        scale_vals = coords_gt_cpu[:, 3].unsqueeze(1).expand(-1, coords_topk_xyz.shape[1])
        coords_topk = torch.cat([coords_topk_xyz, scale_vals.unsqueeze(-1)], dim=-1).to(self.trainer.device)
        self.evaluate_and_report(
            coords_topk,
            coords_gt_cpu,
            tag="Eval_pred_pdf (wo eval scale)",
            dist_lambda=self.eval_cfg["dist_lambda"],
            rot_th=self.eval_cfg["rot_th"],
            scale_ratio_th=None,
            scale_select_mode=None,
        )
        return coords_topk, scores_topk

    def apply_histogram_filter(self, pred_pdf_3d_all, coords_gt_all, n_coarse_3d):
        from util_core_histogram_filter_3d import HistogramFilter3D

        pred_pdf_3d_shaped = pred_pdf_3d_all.to(self.trainer.device).reshape(-1, *n_coarse_3d)
        raw_diff = torch.diff(coords_gt_all[:, 2])
        diff_rot_rad = (raw_diff + torch.pi) % (2 * torch.pi) - torch.pi
        histfilter = HistogramFilter3D(
            H=n_coarse_3d[0],
            W=n_coarse_3d[1],
            O=n_coarse_3d[2],
            device=pred_pdf_3d_shaped.device,
        )
        pred_pdf_3d_hist = pred_pdf_3d_shaped.permute(0, 3, 1, 2)
        preds_filtered = []
        histfilter.belief = histfilter.belief * pred_pdf_3d_hist[0:1]
        preds_filtered.append(histfilter.belief.clone())
        for i in range(diff_rot_rad.shape[0]):
            if i == diff_rot_rad.shape[0]:
                break
            histfilter.predict(
                move_rot=diff_rot_rad[i],
                noise_std_rot=30 / 180 * torch.pi,
                direction_aware=False,
                noise_std_xy=0.65,
                xy_k_size=5,
            )
            histfilter.update(pred_pdf_3d_hist[i + 1:i + 2], alpha=0.25)
            preds_filtered.append(histfilter.belief.clone())
        return torch.cat(preds_filtered).permute(0, 2, 3, 1)



# Backward-compatible alias. Existing callers can keep importing the old name.
Stage3FineLocEvaluator = Stage3FineLocManager
