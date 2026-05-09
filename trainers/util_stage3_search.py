import time
import numpy as np
import torch
import torch.nn.functional as TF
import tqdm

from trainers.util_stage3_loc_manager import Stage3FineLocManager
from trainers.util_stage3_multi_stage_refiner import (
    BatchedMultiStartEvolutionSeedCloudRefiner,
    EvoTorchFinalModeOptimizer,
    GradientTopKOptimizer,
    LocalSeedCloudBuilder,
    ModeState,
    PassthroughModeDeduper,
    SeedModeSearchConfig,
    SeedModeSearchPipeline,
    SeedRegionConfig,
    Stage3CMAConfig,
    Stage4GradConfig,
    TopNSeedScreening,
)

def _opt_coords_topN(self, coords_topN, feat_q, n_step=200,lr=1e-5):
        """
        对TopN候选坐标进行梯度优化重排序

        Args:
            coords_topN: [B, N, 4] - TopN候选坐标 (nr, nc, rot, scale)
            feat_q: [B, C] - query特征
            n_step: int - 优化步数

        Returns:
            coords_sorted: [B, N, 4] - 按优化后的loss排序的坐标
        """
        import math

        # Detach feat_q，因为我们只优化坐标，不需要对query特征求导
        feat_q = feat_q.detach().to(self.device)
        coords_opted_topN = []
        loss_topN = []

        # 临时设置为train模式以启用梯度计算图构建（即使参数是冻结的）
        # 这是必需的，因为eval()模式可能会阻止梯度流动到输入
        grid_was_training = self.grid.training
        grid_mlp_was_training = self.grid_mlp.training
        pos_encoder_was_training = self.pos_encoder_grid.training

        self.grid.train()
        self.grid_mlp.train()
        self.pos_encoder_grid.train()

        for id in range(coords_topN.shape[1]):
            coords_init = coords_topN[:, id, :].clone().detach()

            # 关键改进：分别创建参数并设置不同学习率
            xy_param = coords_init[:, :2].clone().requires_grad_(True)
            rot_param = coords_init[:, 2:3].clone().requires_grad_(True)
            scale_param = coords_init[:, 3:4].clone().requires_grad_(True)

            # 根据各维度的数值范围设置学习率
            optimizer = torch.optim.Adam([
                {'params': [xy_param], 'lr': lr},
                {'params': [rot_param], 'lr': lr*0.5},
                {'params': [scale_param], 'lr': lr}
            ], lr=1e-4)  # 默认学习率（会被参数组覆盖）

            # 可选：学习率调度
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_step, eta_min=1e-6
            )

            for i in range(n_step):
                optimizer.zero_grad()

                # 关键修改：将 enable_grad 提早开启，覆盖参数的使用起点
                with torch.enable_grad():
                    # 1. 组合坐标 (必须在 enable_grad 下进行，才能追踪到 xy_param)
                    coords2opt = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                    # 2. 转换为6D坐标
                    coords_6d = self.coord_normer.raw_to_net(coords2opt, append_linear_rot=True)

                    # 3. 提取特征与前向传播
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feat_ref_raw = self._get_feats_fm_grid(grid_input)


                    coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
                    feat_ref = self.grid_mlp(feat_ref_raw, coords_encoded)
                    feat_ref = TF.normalize(feat_ref, dim=-1, p=2)

                    # 4. 计算 Loss
                    loss = TF.mse_loss(feat_q, feat_ref)

                    # 5. 反向传播
                    loss.backward()
                # 优化步 (可以在 enable_grad 外，也可以在内，通常在外)
                # 可选：梯度裁剪
                torch.nn.utils.clip_grad_norm_([xy_param, rot_param, scale_param], max_norm=1.0)

                optimizer.step()
                scheduler.step()

                if i % 50 == 0:  # 减少打印频率
                    print(f"  候选 {id + 1}/{coords_topN.shape[1]}, Step {i}/{n_step}, Loss: {loss.item():.5f}")

            # 保存最终结果
            with torch.no_grad():
                coords_final = torch.cat([xy_param, rot_param, scale_param], dim=-1)

                # 计算最终loss
                coords_6d_final = self.coord_normer.raw_to_net(coords_final, append_linear_rot=True)
                grid_input_final = torch.cat([coords_6d_final[:, :2], coords_6d_final[:, -1:]], dim=-1)
                feat_ref_final_raw = self._get_feats_fm_grid(grid_input_final)
                coords_encoded_final = self.pos_encoder_grid(coords_6d_final[:, :5])
                feat_ref_final = self.grid_mlp(feat_ref_final_raw, coords_encoded_final)
                feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)

                final_loss = TF.mse_loss(
                    feat_q, feat_ref_final, reduction='none'
                ).mean(dim=-1)

                loss_topN.append(final_loss)
                coords_opted_topN.append(coords_final.detach())

        coords_opted_topN = torch.stack(coords_opted_topN, dim=1)
        loss_topN = torch.stack(loss_topN, dim=1)

        # 候选重排序
        sorted_indices = loss_topN.argsort(dim=1)
        coords_sorted = coords_opted_topN[
            torch.arange(coords_opted_topN.shape[0]).unsqueeze(1),
            sorted_indices
        ]

        # 恢复模型的原始状态
        if not grid_was_training:
            self.grid.eval()
        if not grid_mlp_was_training:
            self.grid_mlp.eval()
        if not pos_encoder_was_training:
            self.pos_encoder_grid.eval()

        return coords_sorted

def _sample_around_candidates(self, coords_centers, grid_dims, space_size=None):
        """
        逻辑2：在给定的中心点周围进行网格采样。

        Args:
            coords_centers: [B, N, 4] 中心点坐标
            grid_dims: tuple (nr, nc, rot, scale) 网格维度
            space_size: list/tuple (Optional) 物理空间大小

        Returns:
            coords_new: [B, N * Points_Per_Center, 4] 采样后的新坐标点
        """
        B, N, _ = coords_centers.shape

        # 1. 展平以适配 Sampler 接口 [B*N, 4]
        centers_flat = coords_centers.reshape(-1, 4)

        # 2. 调用 Sampler
        # 输出: [B*N, Points_Per_Center, 4]
        coords_sampled = self.subspace_sampler.sample_grid_around_coords_gpu(
            center_coords=centers_flat,
            grid_dims=grid_dims,
            sample_space_size=space_size,
            device=self.device
        )

        points_per_center = coords_sampled.shape[1]

        # 3. 恢复 Batch 维度
        # [B*N, Points, 4] -> [B, N*Points, 4]
        # 这里的 reshape 实际上把每个中心点生成的点铺平了
        coords_new = coords_sampled.reshape(B, N * points_per_center, 4)

        return coords_new

def _build_cma_init_seeds(
            self,
            l0_prob_q,
            coords_candidates_flat,
            n_coarse,
            topk_seed,
            scale_select_mode="argmax",
            selection_space="3d",
    ):
        """
        Build initial L0 seed modes from a coarse 4D probability volume.

        2D strategy:
        1. Integrate the 4D probability volume over rot and scale to get a
           2D spatial score for each (nr, nc) cell.
        2. Estimate a continuous rotation per cell using the circular mean of
           the rot marginal.
        3. Estimate a continuous scale per cell using the configured scale
           resolver.
        4. Select Top-N cells from the 2D score volume.

        Legacy 3D strategy is available with selection_space='3d':
        marginalize scale, select Top-N (nr, nc, rot) anchors, then resolve
        scale inside each selected 3D anchor.
        """
        n_nr, n_nc, n_rot, n_scale = [int(x) for x in n_coarse]
        selection_space = str(selection_space).strip().lower()
        if selection_space in ("3d_anchor", "3d_voxel"):
            selection_space = "3d"
        if selection_space in ("2d_cell", "spatial", "nr_nc"):
            selection_space = "2d"
        if selection_space not in ("2d", "3d"):
            raise ValueError(f"selection_space must be '2d' or '3d', got {selection_space}")

        n_anchor = n_nr * n_nc if selection_space == "2d" else n_nr * n_nc * n_rot
        topk_seed = min(int(topk_seed), n_anchor)
        if topk_seed <= 0:
            raise ValueError("topk_seed must be > 0.")

        scale_select_mode = str(scale_select_mode).strip().lower()
        if scale_select_mode not in ("argmax", "log_expectation"):
            raise ValueError(
                f"scale_select_mode must be 'argmax' or 'log_expectation', got {scale_select_mode}"
            )

        l0_prob_q_4d = l0_prob_q.reshape(n_nr, n_nc, n_rot, n_scale)
        coords_candidates_4d = coords_candidates_flat.reshape(n_nr, n_nc, n_rot, n_scale, 4)

        if selection_space == "2d":
            cell_scores_2d = l0_prob_q_4d.sum(dim=(2, 3)).reshape(-1)
            seed_scores, cell_idx_2d = torch.topk(
                cell_scores_2d,
                k=topk_seed,
                dim=-1,
                largest=True,
            )

            probs_by_cell = l0_prob_q_4d.reshape(n_nr * n_nc, n_rot, n_scale).index_select(0, cell_idx_2d)
            coords_by_cell = coords_candidates_4d.reshape(n_nr * n_nc, n_rot, n_scale, 4).index_select(0, cell_idx_2d)

            rot_probs = probs_by_cell.sum(dim=-1)
            rot_weights = rot_probs / (rot_probs.sum(dim=-1, keepdim=True) + 1e-8)
            rot_vals = coords_by_cell[:, :, 0, 2]
            rot_sin = torch.sum(rot_weights * torch.sin(rot_vals), dim=-1)
            rot_cos = torch.sum(rot_weights * torch.cos(rot_vals), dim=-1)
            rot_hat = torch.atan2(rot_sin, rot_cos)

            scale_probs = probs_by_cell.sum(dim=1)
            scale_vals = coords_by_cell[:, 0, :, 3].clamp(min=1e-6)
            if scale_select_mode == "argmax":
                best_scale_idx = torch.argmax(scale_probs, dim=-1)
                row_idx = torch.arange(topk_seed, device=coords_candidates_flat.device)
                scale_hat = scale_vals[row_idx, best_scale_idx]
            else:
                scale_weights = scale_probs / (scale_probs.sum(dim=-1, keepdim=True) + 1e-8)
                scale_log_hat = torch.sum(scale_weights * torch.log(scale_vals), dim=-1)
                scale_hat = torch.exp(scale_log_hat)

            seed_coords = coords_by_cell[:, 0, 0, :].clone()
            seed_coords[:, 2] = rot_hat
            seed_coords[:, 3] = scale_hat
            sort_idx = torch.argsort(seed_scores, descending=True)
            return seed_coords.index_select(0, sort_idx), seed_scores.index_select(0, sort_idx)

        anchor_scores_3d = l0_prob_q_4d.sum(dim=-1).reshape(-1)
        anchor_scores_sel, anchor_idx_3d = torch.topk(
            anchor_scores_3d,
            k=topk_seed,
            dim=-1,
            largest=True,
        )

        probs_by_anchor = l0_prob_q_4d.reshape(-1, n_scale).index_select(0, anchor_idx_3d)
        coords_by_anchor = coords_candidates_flat.reshape(n_nr * n_nc * n_rot, n_scale, 4).index_select(0, anchor_idx_3d)

        if scale_select_mode == "argmax":
            best_scale_idx = torch.argmax(probs_by_anchor, dim=-1)
            row_idx = torch.arange(topk_seed, device=coords_candidates_flat.device)
            seed_coords = coords_by_anchor[row_idx, best_scale_idx]
            seed_scores = probs_by_anchor[row_idx, best_scale_idx]
        else:
            weights = probs_by_anchor / (probs_by_anchor.sum(dim=-1, keepdim=True) + 1e-8)
            scale_vals = coords_by_anchor[..., 3].clamp(min=1e-6)
            scale_log_hat = torch.sum(weights * torch.log(scale_vals), dim=-1)
            seed_coords = coords_by_anchor[:, 0, :].clone()
            seed_coords[:, 3] = torch.exp(scale_log_hat)
            seed_scores = anchor_scores_sel

        sort_idx = torch.argsort(seed_scores, descending=True)
        seed_coords = seed_coords.index_select(0, sort_idx)
        seed_scores = seed_scores.index_select(0, sort_idx)
        return seed_coords, seed_scores

def _pack_mode_states_for_eval(self, mode_lists, topk=None):
        """
        将变长 mode 列表打包成 [B, K, 4] / [B, K]，便于统一评估。
        """
        if len(mode_lists) == 0:
            empty_coords = torch.zeros((0, 0, 4), device=self.device, dtype=torch.float32)
            empty_scores = torch.zeros((0, 0), device=self.device, dtype=torch.float32)
            return empty_coords, empty_scores

        if topk is None:
            topk = max(1, max(len(modes) for modes in mode_lists))
        topk = max(1, int(topk))

        coords_all = []
        scores_all = []
        for modes in mode_lists:
            if len(modes) == 0:
                coords_q = torch.zeros((topk, 4), device=self.device, dtype=torch.float32)
                scores_q = torch.full((topk,), -1e9, device=self.device, dtype=torch.float32)
                coords_all.append(coords_q)
                scores_all.append(scores_q)
                continue

            modes_sorted = sorted(
                modes,
                key=lambda m: float(m.best_score if m.best_score is not None else (m.latest_metric or -1e9)),
                reverse=True,
            )[:topk]
            coords_q = torch.stack(
                [
                    (m.best_coord_raw if m.best_coord_raw is not None else m.center_raw).to(self.device)
                    for m in modes_sorted
                ],
                dim=0,
            )
            scores_q = torch.tensor(
                [
                    float(m.best_score if m.best_score is not None else (m.latest_metric or -1e9))
                    for m in modes_sorted
                ],
                device=self.device,
                dtype=torch.float32,
            )
            if coords_q.shape[0] < topk:
                pad_n = topk - coords_q.shape[0]
                coords_q = torch.cat([coords_q, coords_q[-1:].expand(pad_n, -1)], dim=0)
                scores_q = torch.cat(
                    [scores_q, torch.full((pad_n,), -1e9, device=self.device, dtype=torch.float32)],
                    dim=0,
                )
            coords_all.append(coords_q)
            scores_all.append(scores_q)

        return torch.stack(coords_all, dim=0), torch.stack(scores_all, dim=0)


def _pack_stage_records_for_eval(
        self,
        query_results,
        stage_name,
        topk=None,
        fallback_coords=None,
        fallback_scores=None,
    ):
        """
        从 query_results.stage_trace 中提取指定 stage 的 Top-K 坐标/分数，并打包成 [B, K, 4] / [B, K]。
        若某个 query 没有该 stage 记录，则使用 fallback。
        """
        if len(query_results) == 0:
            empty_coords = torch.zeros((0, 0, 4), device=self.device, dtype=torch.float32)
            empty_scores = torch.zeros((0, 0), device=self.device, dtype=torch.float32)
            return empty_coords, empty_scores

        stage_name = str(stage_name).strip().lower()
        resolved_records = []
        max_topk = 0
        for query_result in query_results:
            stage_trace = getattr(query_result, "stage_trace", None)
            stage_records = getattr(stage_trace, "stage_records", None) if stage_trace is not None else None
            matched = None
            if stage_records is not None:
                matched_candidates = [
                    record
                    for record in stage_records
                    if str(getattr(record, "stage_name", "")).strip().lower() == stage_name
                    and getattr(record, "coords_topk_raw", None) is not None
                    and int(record.coords_topk_raw.shape[0]) > 0
                ]
                if len(matched_candidates) > 0:
                    matched = max(matched_candidates, key=lambda record: int(getattr(record, "stage_id", 0)))
                    max_topk = max(max_topk, int(matched.coords_topk_raw.shape[0]))
            resolved_records.append(matched)

        if topk is None:
            if max_topk > 0:
                topk = max_topk
            elif fallback_coords is not None:
                topk = int(fallback_coords.shape[1]) if fallback_coords.ndim == 3 else 1
            else:
                topk = 1
        topk = max(1, int(topk))

        coords_all = []
        scores_all = []
        for q_idx, record in enumerate(resolved_records):
            if record is not None:
                coords_q = record.coords_topk_raw.detach().to(device=self.device, dtype=torch.float32)
                scores_q = record.scores_topk.detach().to(device=self.device, dtype=torch.float32)
            elif fallback_coords is not None:
                coords_q = fallback_coords[q_idx].detach().to(device=self.device, dtype=torch.float32)
                if coords_q.ndim == 1:
                    coords_q = coords_q.unsqueeze(0)
                if fallback_scores is not None:
                    scores_q = fallback_scores[q_idx].detach().to(device=self.device, dtype=torch.float32)
                    if scores_q.ndim == 0:
                        scores_q = scores_q.unsqueeze(0)
                else:
                    scores_q = torch.full((coords_q.shape[0],), -1e9, device=self.device, dtype=torch.float32)
            else:
                coords_q = torch.zeros((0, 4), device=self.device, dtype=torch.float32)
                scores_q = torch.zeros((0,), device=self.device, dtype=torch.float32)

            coords_q = coords_q[:topk]
            scores_q = scores_q[:topk]
            if coords_q.shape[0] == 0:
                coords_q = torch.zeros((topk, 4), device=self.device, dtype=torch.float32)
                scores_q = torch.full((topk,), -1e9, device=self.device, dtype=torch.float32)
            elif coords_q.shape[0] < topk:
                pad_n = topk - coords_q.shape[0]
                coords_q = torch.cat([coords_q, coords_q[-1:].expand(pad_n, -1)], dim=0)
                scores_q = torch.cat(
                    [scores_q, torch.full((pad_n,), -1e9, device=self.device, dtype=torch.float32)],
                    dim=0,
                )

            coords_all.append(coords_q)
            scores_all.append(scores_q)

        return torch.stack(coords_all, dim=0), torch.stack(scores_all, dim=0)


def _sort_topk_metric_keys(metric_keys):
        def _key_to_int(metric_key):
            text = str(metric_key)
            if text.startswith("top") and text.endswith("_acc"):
                middle = text[3:-4]
                if middle.isdigit():
                    return int(middle)
            return 10**9

        return sorted(metric_keys, key=lambda key: (_key_to_int(key), str(key)))


def _resolve_l0_seed_topk(l0_topN, l0_ratio, stage1_mode_input_max, n_anchor_l0):
        n_anchor_l0 = int(n_anchor_l0)
        if n_anchor_l0 <= 0:
            raise ValueError("n_anchor_l0 must be > 0.")

        if l0_topN is not None:
            requested = int(l0_topN)
            if requested <= 0:
                raise ValueError("l0_topN must be > 0 when provided.")
            source = "l0_topN"
            ratio_value = None if l0_ratio is None else float(l0_ratio)
        else:
            if l0_ratio is None:
                raise ValueError("l0_topN=None requires l0_ratio to be set.")
            ratio_value = float(l0_ratio)
            if ratio_value <= 0.0 or ratio_value > 1.0:
                raise ValueError("l0_ratio must be in (0, 1] when l0_topN is None.")
            requested = int(np.ceil(ratio_value * float(n_anchor_l0)))
            source = "l0_ratio"

        requested_before_cap = int(requested)
        max_cap = None
        if stage1_mode_input_max is not None:
            max_cap = int(stage1_mode_input_max)
            if max_cap <= 0:
                raise ValueError("stage1_mode_input_max must be > 0 when provided.")
            requested = min(int(requested), max_cap)

        resolved = max(1, min(int(requested), n_anchor_l0))
        info = {
            "source": source,
            "n_anchor_l0": int(n_anchor_l0),
            "l0_topN": None if l0_topN is None else int(l0_topN),
            "l0_ratio": ratio_value,
            "stage1_mode_input_max": max_cap,
            "requested_before_cap": int(requested_before_cap),
            "requested_after_cap": int(requested),
            "resolved_topk_seed": int(resolved),
            "clipped_by_total": bool(int(requested) > n_anchor_l0),
            "l0_ratio_ignored": bool(l0_topN is not None and l0_ratio is not None),
        }
        return int(resolved), info


def _build_progressive_recall_delta_report(stage_reports):
        """
        Build a same-threshold recall comparison across coarse, seed-mode, and final stages.
        """
        criteria = (
            ("dist_recall", "Dist"),
            ("dist_rot_recall", "Dist+Rot"),
            ("dist_rot_scale_recall", "Dist+Rot+Scale"),
        )
        stage_names = ("coarse_retrieval", "seed_mode_init", "seed_mode_final")
        stage_labels = {
            "coarse_retrieval": "Coarse",
            "seed_mode_init": "SeedInit",
            "seed_mode_final": "Final",
        }

        def _progressive(stage_name, criterion_key):
            report = stage_reports.get(stage_name, {})
            if not isinstance(report, dict):
                return {}
            progressive = report.get("progressive_acc_metrics", {})
            if not isinstance(progressive, dict):
                return {}
            metrics = progressive.get(criterion_key, {})
            return metrics if isinstance(metrics, dict) else {}

        payload = {
            "baseline": "coarse_retrieval",
            "stages": list(stage_names),
            "stage_labels": dict(stage_labels),
            "criteria": {},
        }

        print("\n" + "=" * 96)
        print("Threshold Progressive Recall Delta (baseline: Coarse-Retrieval)")
        print("=" * 96)
        print(f"{'Criterion':<16} {'K':>8} {'Coarse':>10} {'SeedInit':>10} {'DeltaInit':>10} {'Final':>10} {'DeltaFinal':>10}")
        print("-" * 96)

        any_row = False
        for criterion_key, criterion_label in criteria:
            metric_sets = {
                stage_name: _progressive(stage_name, criterion_key)
                for stage_name in stage_names
            }
            common_keys = set(metric_sets["coarse_retrieval"].keys())
            for stage_name in stage_names[1:]:
                common_keys &= set(metric_sets[stage_name].keys())
            sorted_keys = _sort_topk_metric_keys(common_keys)

            criterion_payload = {}
            for metric_key in sorted_keys:
                coarse = float(metric_sets["coarse_retrieval"][metric_key])
                seed_init = float(metric_sets["seed_mode_init"][metric_key])
                final = float(metric_sets["seed_mode_final"][metric_key])
                init_delta = seed_init - coarse
                final_delta = final - coarse
                criterion_payload[str(metric_key)] = {
                    "coarse_retrieval": coarse,
                    "seed_mode_init": seed_init,
                    "seed_mode_init_delta": init_delta,
                    "seed_mode_final": final,
                    "seed_mode_final_delta": final_delta,
                }
                print(
                    f"{criterion_label:<16} {str(metric_key):>8} "
                    f"{coarse:10.2f} {seed_init:10.2f} {init_delta:+10.2f} "
                    f"{final:10.2f} {final_delta:+10.2f}"
                )
                any_row = True

            payload["criteria"][criterion_key] = criterion_payload

        if not any_row:
            print("[WARN] No common progressive recall keys found for coarse/init/final reports.")
        print("=" * 96 + "\n")

        return payload


def _test_3d_fine_accuracy_hds(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=False,
            enable_filter=False,
            chunk_size=2048,
            n_bins_4d=None,
            n_bins_scale_mode="linear",
            eval_thresh_cfg=None,
            scale_select_mode="log_expectation",

            ge_prob_mode="ingp",
            ge_top_ratio_rho0=None,
            ge_topk_seed=None,
            ge_max_anchors_k0=None,
            ge_seed_selection_space="3d",

            pc_score_mode="ingp",
            pc_particles_per_round=(32,),
            pc_radius_scale_per_round=(1.0,),
            pc_local_sample_method="sobol_deterministic",
            pc_elite_ratio_rho=0.25,
            pc_population_survival_ratio_per_round=(1.0,),
            pc_min_surviving_populations=1,
            pc_enable_scale_sampling=True,
            pc_survive_stand="best",
            pc_move_stand="elite_sum",

            lr_score_mode="ingp",
            lr_enable=True,
            lr_variant="Sep-CMA",
            lr_init_sigma=None,
            lr_popsize=32,
            lr_iters=8,
            lr_enable_early_stop=False,
            lr_early_stop_patience=3,
            lr_max_input_modes=16,
            lr_optimize_scale=True,
            lr_enable_competition=False,
            lr_competition_interval=2,
            lr_survival_ratio=0.5,
            lr_min_surviving_modes=1,
            lr_elite_ratio=0.25,
            rerank_per_mode_after_lr=False,
            debug_stage_timing=True,
    ):
        """Hierarchical Evolutionary Search interface using paper-stage names.

        Naming:
        - GE: Global Exploration
        - PC: Population Contraction
        - LR: Local Refinement
        """
        dict_res = _test_3d_fine_accuracy_seed_mode_CMA_ES(
            self,
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            temperature=temperature,
            shuffle=shuffle,
            save_pred_pdf=save_pred_pdf,
            enable_filter=enable_filter,
            chunk_size=chunk_size,
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode=n_bins_scale_mode,
            l0_prob_mode=ge_prob_mode,
            l0_topN=ge_topk_seed,
            l0_ratio=ge_top_ratio_rho0,
            stage1_mode_input_max=ge_max_anchors_k0,
            eval_thresh_cfg=eval_thresh_cfg,
            scale_select_mode=scale_select_mode,
            l0_seed_selection_space=ge_seed_selection_space,
            stage1_alpha=1.0,
            stage1_prob_mode=pc_score_mode,
            stage1_samples_per_round=pc_particles_per_round,
            stage1_radius_scale_per_round=pc_radius_scale_per_round,
            stage1_local_sample_method=pc_local_sample_method,
            stage1_elite_ratio=pc_elite_ratio_rho,
            stage1_survival_ratio_per_round=pc_population_survival_ratio_per_round,
            stage1_min_surviving_clouds=pc_min_surviving_populations,
            stage1_enable_scale_sampling=pc_enable_scale_sampling,
            stage1_5_refiner_mode="multi_start_es",
            stage1_survive_stand=pc_survive_stand,
            stage1_move_stand=pc_move_stand,
            cma_prob_mode=lr_score_mode,
            cma_optimize_scale=lr_optimize_scale,
            cma_variant=lr_variant,
            cma_init_sigma_manual=lr_init_sigma,
            cma_popsize=lr_popsize,
            cma_iters=lr_iters,
            cma_enable_early_stop=lr_enable_early_stop,
            cma_early_stop_patience=lr_early_stop_patience,
            cma_enable_competition=lr_enable_competition,
            cma_competition_interval=lr_competition_interval,
            cma_survival_ratio=lr_survival_ratio,
            cma_min_surviving_modes=lr_min_surviving_modes,
            cma_elite_ratio=lr_elite_ratio,
            cma_max_input_mode=lr_max_input_modes,
            rerank_per_mode_after_stage3=rerank_per_mode_after_lr,
            stage3_enable=lr_enable,
            stage4_enable=False,
            debug_stage_timing=debug_stage_timing,
            refine_loc=lr_enable,
        )
        dict_res["hds_interface_config"] = {
            "ge_prob_mode": str(ge_prob_mode),
            "ge_top_ratio_rho0": None if ge_top_ratio_rho0 is None else float(ge_top_ratio_rho0),
            "ge_topk_seed": None if ge_topk_seed is None else int(ge_topk_seed),
            "ge_max_anchors_k0": None if ge_max_anchors_k0 is None else int(ge_max_anchors_k0),
            "ge_seed_selection_space": str(ge_seed_selection_space),
            "pc_score_mode": str(pc_score_mode),
            "pc_particles_per_round": [int(x) for x in self._ensure_param_sequence(pc_particles_per_round, int)],
            "pc_radius_scale_per_round": [float(x) for x in self._ensure_param_sequence(pc_radius_scale_per_round, float)],
            "pc_local_sample_method": str(pc_local_sample_method),
            "pc_elite_ratio_rho": float(pc_elite_ratio_rho),
            "pc_population_survival_ratio_per_round": [
                float(x) for x in self._ensure_param_sequence(pc_population_survival_ratio_per_round, float)
            ],
            "pc_min_surviving_populations": int(pc_min_surviving_populations),
            "pc_enable_scale_sampling": bool(pc_enable_scale_sampling),
            "pc_survive_stand": str(pc_survive_stand),
            "pc_move_stand": str(pc_move_stand),
            "lr_score_mode": str(lr_score_mode),
            "lr_enable": bool(lr_enable),
            "lr_variant": str(lr_variant),
            "lr_init_sigma": None if lr_init_sigma is None else float(lr_init_sigma),
            "lr_popsize": int(lr_popsize),
            "lr_iters": int(lr_iters),
            "lr_enable_early_stop": bool(lr_enable_early_stop),
            "lr_early_stop_patience": int(lr_early_stop_patience),
            "lr_max_input_modes": int(lr_max_input_modes),
            "lr_optimize_scale": bool(lr_optimize_scale),
            "lr_enable_competition": bool(lr_enable_competition),
            "lr_competition_interval": int(lr_competition_interval),
            "lr_survival_ratio": float(lr_survival_ratio),
            "lr_min_surviving_modes": int(lr_min_surviving_modes),
            "lr_elite_ratio": float(lr_elite_ratio),
        }
        return dict_res


def _test_3d_fine_accuracy_seed_mode_CMA_ES(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=False,
            enable_filter=False,
            chunk_size=2048,
            n_bins_4d=None,
            n_bins_scale_mode="log",
            l0_prob_mode='ingp',
            l0_topN=64,
            l0_ratio=None,
            stage1_mode_input_max=None,
            eval_thresh_cfg=None,
            scale_select_mode='argmax',
            l0_seed_selection_space="2d",
            stage1_alpha=1.0,
            stage1_prob_mode='ingp',
            stage1_samples_per_round=(32,),
            stage1_radius_scale_per_round=(1.0,),
            stage1_local_sample_method="sobol",
            stage1_elite_ratio=0.25,
            stage1_survival_ratio_per_round=(1.0,),
            stage1_min_surviving_clouds=1,
            stage1_enable_scale_sampling=True,
            stage1_5_refiner_mode="multi_start_es",
            stage1_survive_stand="best",
            stage1_move_stand="elite_sum",
            cma_prob_mode='ingp',
            cma_optimize_scale=True,
            cma_variant='CMA',
            cma_init_sigma_manual=None,
            cma_popsize=32,
            cma_iters=8,
            cma_enable_early_stop=False,
            cma_early_stop_patience=3,
            cma_enable_competition=False,
            cma_competition_interval=2,
            cma_survival_ratio=0.5,
            cma_min_surviving_modes=1,
            cma_elite_ratio=0.25,
            cma_max_input_mode=16,
            rerank_per_mode_after_stage3=False,
            stage3_enable=None,
            stage4_enable=False,
            stage4_prob_mode='ingp',
            stage4_input_stage='latest',
            stage4_topk_input=32,
            stage4_opt_space='linear',
            stage4_n_steps=100,
            stage4_lr_xy=1e-5,
            stage4_lr_rot=5e-6,
            stage4_lr_scale=1e-5,
            stage4_optimize_scale=True,
            stage4_verbose=False,
            stage4_log_interval=10,
            debug_stage_timing=True,
            refine_loc=True,
    ):
        """
        新版多阶段评估：
        Stage 1 seed screening + local seed-cloud
        Stage 1.5 relocation
        Stage 3 optional CMA-ES refinement
        Stage 4 optional gradient refinement

        参数说明:
            n_samples (int | None):
                评估多少个 query。None 表示全量评估。
            use_train_uav (bool):
                True 使用训练集 UAV，False 使用测试集 UAV。
            temperature (float):
                Projector 路径 softmax 温度，仅影响 projector 概率分布锐度。
            shuffle (bool):
                DataLoader 是否打乱。
            save_pred_pdf (bool):
                是否保存 coarse 3D 概率分布。
            enable_filter (bool):
                是否对 coarse 3D 结果应用时序 HistogramFilter，仅影响 coarse 报告。
            chunk_size (int):
                候选点评分时的前向分块大小，显存不足时需要调小。
            n_bins_4d (list[int] | None):
                coarse 搜索网格分辨率 [NR, NC, Rot, Scale]。
                None 时使用 sampler 默认 coarse 网格。
            n_bins_scale_mode (str):
                自定义网格时 scale 维的离散方式。

            l0_prob_mode (str):
                Stage 0 / L0 粗筛评分模式，可选 {'ingp','projector','product'}。
            l0_topN (int | None):
                从 coarse seeds 中保留多少个 seed 进入 Stage 1。
                若为 None，则改用 l0_ratio * coarse 3D anchor 总数解析。
            l0_ratio (float | None):
                l0_topN 为 None 时使用，取值范围 (0, 1]。
            stage1_mode_input_max (int | None):
                对解析后的 Stage 1 / Stage 1.5 输入 seed 数量做上限裁剪。
            eval_thresh_cfg (dict | None):
                最终评估阈值覆盖，例如 {"dist_lambda":1.1,"rot_th":11.0,"scale_ratio_th":1.15}。
                也支持显式传 {"dist_th":..., ...} 或 {"dist_th_meter":..., ...}。
            scale_select_mode (str):
                coarse 4D -> 初始 seed 时如何解析 scale：
                - 'argmax': 取最佳离散 scale
                - 'log_expectation': 对 log-scale 做加权平均
            l0_seed_selection_space (str):
                L0 seed topN 选择空间，可选 {'2d','3d'}。
                '2d' 先对 rot/scale 积分得到 2D 概率体，再估计连续 rot/scale；
                '3d' 为旧逻辑，先对 scale 积分再选 (nr,nc,rot)。

            stage1_alpha (float):
                Stage 1 局部采样半径系数，实际半径为 r_d = alpha * Delta_d / 2。
            stage1_prob_mode (str):
                Stage 1 seed-cloud 内局部评估使用的评分模式，可选 {'ingp','projector','product'}。
            stage1_samples_per_round (tuple[int, ...]):
                Stage 1 每轮每个 seed-cloud 重采样多少个点。
            stage1_radius_scale_per_round (tuple[float, ...]):
                Stage 1 每轮采样范围系数，相对基础半径 r_d = alpha * Delta_d / 2 缩放。
            stage1_local_sample_method (str):
                Stage 1 局部采样方式，当前支持 'sobol' / 'sobol_deterministic' / 'uniform'。
            stage1_elite_ratio (float):
                Stage 1 每轮取 top-q 比例 elite 点做聚合中心更新。
            stage1_survival_ratio_per_round (tuple[float, ...]):
                Stage 1 每轮保留多少比例的 seed-cloud，可按轮变化。
            stage1_min_surviving_clouds (int):
                Stage 1 每轮至少保留多少个 seed-cloud。
            stage1_enable_scale_sampling (bool):
                若为 True，则 Stage 1 重采样包含 scale 维；False 时固定当前中心的 scale。
            stage1_5_refiner_mode (str):
                Stage 1.5 refiner. Only 'multi_start_es' is kept in the clean pipeline.
                multi_start_es 为 batched 多启动进化策略，不更新协方差/sigma。
            stage1_survive_stand (str):
                multi_start_es 中 mode survival 排序标准，可选 {'best','elite_sum'}。
            stage1_move_stand (str):
                multi_start_es 中下一轮采样中心来源，可选 {'best','elite_sum'}。

            cma_prob_mode (str):
                Stage 3 / CMA-ES 精修时的评分模式，可选 {'ingp','projector','product'}。
            cma_optimize_scale (bool):
                是否让 CMA 同时优化 scale 维。False 时仅优化 [nr, nc, rot]，scale 固定为输入 mode 的 scale。
            cma_variant (str):
                CMA 后端类型，可选 {'CMA','Sep-CMA'}。
            cma_init_sigma_manual (float | None):
                手动指定 Stage 3 的初始 sigma。
                若不为 None，则优先级最高。
            cma_popsize (int):
                每个 mode 的 CMA 每代种群大小。
            cma_iters (int):
                每个 mode 的 CMA 迭代代数。
            cma_enable_early_stop (bool):
                是否启用 CMA 早停。
            cma_early_stop_patience (int):
                若启用早停，连续多少代无提升则停止。
            cma_enable_competition (bool):
                是否启用 Stage 3 的竞争淘汰机制。
            cma_competition_interval (int):
                在 warmup 之后，每隔多少代做一次 mode 间比较和淘汰。
            cma_survival_ratio (float):
                每轮 CMA 竞争后保留多少比例的 mode。
            cma_min_surviving_modes (int):
                每轮 CMA 竞争后至少保留多少个 mode。
            cma_elite_ratio (float):
                用当前采样点中 top-q 比例 elite 样本计算竞争指标。
            cma_max_input_mode (int):
                最多多少个 surviving modes 会进入最终 CMA-ES。
            rerank_per_mode_after_stage3 (bool):
                True 时，Stage 3 结束后对每个 mode 的两个候选坐标做 winner 比较：
                {Stage3-CMA best, Stage1.5 best}。
                若 stage1_prob_mode 与 cma_prob_mode 一致，则直接复用两者已保存分数；
                否则用 Stage 3 的评分函数对两者重新打分。
            stage3_enable (bool | None):
                是否启用 Stage 3。None 时回退到 refine_loc。
            stage4_enable (bool):
                是否启用 Stage 4 梯度优化。
            stage4_prob_mode (str):
                Stage 4 最终输出分数使用的评估函数，可选 {'ingp','projector','product'}。
            stage4_input_stage (str):
                Stage 4 输入来源，默认取最新有效阶段。
            stage4_topk_input (int):
                Stage 4 最多接收多少个 topK 坐标。
            stage4_opt_space (str):
                Stage 4 在哪种坐标空间中做梯度优化，可选 {'linear','raw'}。
            stage4_n_steps (int):
                Stage 4 梯度优化步数。
            stage4_lr_xy / stage4_lr_rot / stage4_lr_scale (float):
                Stage 4 对不同维度使用的学习率。
            stage4_optimize_scale (bool):
                Stage 4 是否优化 scale 维。
            stage4_verbose (bool):
                是否打印 Stage 4 内部优化日志。
            stage4_log_interval (int):
                若启用 Stage 4 日志，每隔多少步打印一次当前 loss mean。
            refine_loc (bool):
                兼容旧参数。stage3_enable 为 None 时，用它决定是否启用 Stage 3。
            debug_stage_timing (bool):
                是否打印 Stage 0/1/3/4 的分段耗时和候选数量，便于定位瓶颈。

        返回:
            dict:
                {
                    "scores_grid": [B, Kg],   # Stage0 coarse retrieval scores
                    "coords_grid": [B, Kg, 4],# Stage0 coarse retrieval coords
                    "scores_mode": [B, Km],   # Stage1.5 mode scores
                    "coords_mode": [B, Km, 4],# Stage1.5 mode coords
                    "scores_evo":  [B, Ke],   # Stage3 CMA-ES final scores
                    "coords_evo":  [B, Ke, 4],# Stage3 CMA-ES final coords
                    "coords_gt":   [B, 4],    # GT 坐标
                    "query_results": list,    # 每个 query 的完整 pipeline 结果
                }
        """
        evaluator = Stage3FineLocManager(
            self,
            temperature=temperature,
            chunk_size=chunk_size,
            eval_thresh_cfg=eval_thresh_cfg,
        )
        from trainers.util_stage3_multi_start_CMAES_by_evotorch import MultiStartCMAESEvoTorchRefiner

        cma_refiner = MultiStartCMAESEvoTorchRefiner(coords_processor=self.coord_normer, device=self.device)

        def _score_pair_fn_for_refiner(query_feats_mxc, coords_raw_mx4):
            with torch.no_grad():
                scores = evaluator.score_candidates(
                    query_feats=query_feats_mxc,
                    coords_batch=coords_raw_mx4.unsqueeze(1),
                    mode=cma_prob_mode,
                    normalize=False,
                )
            scores = scores.reshape(-1)
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
            return scores

        l0_prob_mode = evaluator.validate_prob_mode("l0_prob_mode", l0_prob_mode)
        cma_prob_mode = evaluator.validate_prob_mode("cma_prob_mode", cma_prob_mode)
        stage4_prob_mode = evaluator.validate_prob_mode("stage4_prob_mode", stage4_prob_mode)
        stage4_opt_space = str(stage4_opt_space).strip().lower()
        if stage4_opt_space not in ("linear", "raw"):
            raise ValueError(f"stage4_opt_space must be 'linear' or 'raw', got {stage4_opt_space}")
        if stage3_enable is None:
            stage3_enable = bool(refine_loc)
        scale_select_mode = str(scale_select_mode).strip().lower()
        if scale_select_mode not in ("argmax", "log_expectation"):
            raise ValueError(f"scale_select_mode must be 'argmax' or 'log_expectation', got {scale_select_mode}")
        l0_seed_selection_space = str(l0_seed_selection_space).strip().lower()
        if l0_seed_selection_space in ("2d_cell", "spatial", "nr_nc"):
            l0_seed_selection_space = "2d"
        if l0_seed_selection_space in ("3d_anchor", "3d_voxel"):
            l0_seed_selection_space = "3d"
        if l0_seed_selection_space not in ("2d", "3d"):
            raise ValueError(f"l0_seed_selection_space must be '2d' or '3d', got {l0_seed_selection_space}")
        stage1_5_refiner_mode = str(stage1_5_refiner_mode).strip().lower()
        if stage1_5_refiner_mode in ("batched_fixed_step_es", "fixed_step", "fixed-step-es", "fixed_step_es", "batched_multi_start_es", "ms_es"):
            stage1_5_refiner_mode = "multi_start_es"
        if stage1_5_refiner_mode != "multi_start_es":
            raise ValueError(f"stage1_5_refiner_mode must be 'multi_start_es', got {stage1_5_refiner_mode}")
        stage1_survive_stand = str(stage1_survive_stand).strip().lower()
        stage1_move_stand = str(stage1_move_stand).strip().lower()
        if stage1_survive_stand not in ("best", "elite_sum"):
            raise ValueError(f"stage1_survive_stand must be 'best' or 'elite_sum', got {stage1_survive_stand}")
        if stage1_move_stand not in ("best", "elite_sum"):
            raise ValueError(f"stage1_move_stand must be 'best' or 'elite_sum', got {stage1_move_stand}")

        print(f"\n{'=' * 60}")
        print("3D细粒度定位测试 (Seed-Mode Search + Optional CMA-ES + Optional Grad)")
        print(f"测试样本数: {n_samples if n_samples is not None else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(
            f"L0 mode: {l0_prob_mode} | "
            f"Stage1 mode: {stage1_prob_mode} | "
            f"Stage3 mode: {cma_prob_mode} | CMA variant: {cma_variant} | "
            f"Stage3 enabled: {bool(stage3_enable)} | "
            f"Stage3 competition: {bool(cma_enable_competition)} | "
            f"Stage4 enabled: {bool(stage4_enable)} | Stage4 mode: {stage4_prob_mode} | "
            f"Stage4 opt space: {stage4_opt_space}"
        )
        print(
            f"Stage1.5 refiner: {stage1_5_refiner_mode} | "
            f"survive_stand: {stage1_survive_stand} | move_stand: {stage1_move_stand}"
        )
        print(f"L0 seed selection space: {l0_seed_selection_space}")
        print(f"{'=' * 60}\n")

        t_stage0 = time.perf_counter()
        coarse = evaluator.run_l0_coarse_retrieval(
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            shuffle=shuffle,
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode=n_bins_scale_mode,
            l0_prob_mode=l0_prob_mode,
            save_pred_pdf=save_pred_pdf,
            save_tag="seed_mode_pipeline",
            report_title="Spatial Classification Results (Seed-Mode Pipeline, wo eval scale)",
            show_progress=not bool(debug_stage_timing),
            progress_desc="SeedMode L0 coarse retrieval",
        )
        if bool(debug_stage_timing):
            print(f"[SeedMode] Stage0-CoarseRetrieval done | n_query={int(coarse['coords_gt_all'].shape[0])} | {time.perf_counter() - t_stage0:.3f}s")
        pred_pdf_3d_all = coarse["pred_pdf_3d_all"]
        q_label_3d_all = coarse["q_label_3d_all"]
        coords_gt_all = coarse["coords_gt_all"]
        feats_vis_all = coarse["feats_vis_all"]
        l0_prob_4d_all = coarse["l0_prob_4d_all"]
        coords_candidates_flat = coarse["coords_candidates_flat"]
        n_coarse = coarse["n_coarse"]
        n_coarse_3d = coarse["n_coarse_3d"]
        spatial_classification_report = coarse.get("spatial_classification_report", None)
        filtered_spatial_classification_report = None

        if enable_filter:
            preds_filtered = evaluator.apply_histogram_filter(
                pred_pdf_3d_all=pred_pdf_3d_all,
                coords_gt_all=coords_gt_all,
                n_coarse_3d=n_coarse_3d,
            )
            filtered_metrics = evaluator.report_spatial_classification(
                pred_pdf_3d_all=preds_filtered.reshape(preds_filtered.shape[0], -1),
                q_label_3d_all=q_label_3d_all,
                title="Spatial Classification Results, Filtered (Seed-Mode Pipeline)",
            )
            filtered_spatial_classification_report = {
                "report_title": "Spatial Classification Results, Filtered (Seed-Mode Pipeline)",
                "metrics": filtered_metrics,
            }

        n_nr, n_nc, n_rot, n_scale = [int(x) for x in n_coarse]
        delta_nr = float(self.coord_normer.nrc_diff[0].item()) / max(1, n_nr)
        delta_nc = float(self.coord_normer.nrc_diff[1].item()) / max(1, n_nc)
        delta_rot = float(2.0 * np.pi) / max(1, n_rot)
        scale_min = float(torch.exp(self.coord_normer.scale_log_min).item())
        scale_max = float(torch.exp(self.coord_normer.scale_log_max).item())
        delta_scale = (scale_max - scale_min) / max(1, n_scale)
        n_anchor_3d = int(np.prod(n_coarse_3d))
        n_anchor_l0 = int(n_nr * n_nc) if l0_seed_selection_space == "2d" else n_anchor_3d
        topk_seed, l0_seed_resolution = _resolve_l0_seed_topk(
            l0_topN=l0_topN,
            l0_ratio=l0_ratio,
            stage1_mode_input_max=stage1_mode_input_max,
            n_anchor_l0=n_anchor_l0,
        )
        l0_seed_resolution["selection_space"] = str(l0_seed_selection_space)
        l0_seed_resolution["n_anchor_3d"] = int(n_anchor_3d)
        print(
            "[SeedMode] Stage1 input seeds: "
            f"{topk_seed}/{n_anchor_l0} | selection_space={l0_seed_selection_space} | "
            f"source={l0_seed_resolution['source']} | "
            f"l0_topN={l0_seed_resolution['l0_topN']} | "
            f"l0_ratio={l0_seed_resolution['l0_ratio']} | "
            f"stage1_mode_input_max={l0_seed_resolution['stage1_mode_input_max']}"
        )
        stage1_bin_sizes_raw = torch.tensor(
            [delta_nr, delta_nc, delta_rot, delta_scale],
            device=self.device,
            dtype=torch.float32,
        )

        pipeline_cfg = SeedModeSearchConfig(
            score_mode_stage1=stage1_prob_mode,
            score_mode_stage3=cma_prob_mode,
            metric_goal="maximize",
            seed_region=SeedRegionConfig(
                n_bins_4d=tuple(int(x) for x in n_coarse),
                topN_seed=int(topk_seed),
                alpha=float(stage1_alpha),
                samples_per_round=self._ensure_param_sequence(stage1_samples_per_round, int),
                radius_scale_per_round=self._ensure_param_sequence(stage1_radius_scale_per_round, float),
                local_sample_method=stage1_local_sample_method,
                elite_ratio=float(stage1_elite_ratio),
                survival_ratio_per_round=self._ensure_param_sequence(stage1_survival_ratio_per_round, float),
                min_surviving_clouds=int(stage1_min_surviving_clouds),
                enable_scale_sampling=bool(stage1_enable_scale_sampling),
            ),
            stage3=Stage3CMAConfig(
                enable=bool(stage3_enable),
                cma_max_input_mode=int(cma_max_input_mode),
                score_mode=cma_prob_mode,
                cma_optimize_scale=bool(cma_optimize_scale),
                cma_variant=cma_variant,
                init_sigma_manual=None if cma_init_sigma_manual is None else float(cma_init_sigma_manual),
                popsize=int(cma_popsize),
                n_iterations=int(cma_iters),
                enable_early_stop=bool(cma_enable_early_stop),
                early_stop_patience=int(cma_early_stop_patience),
                enable_competition=bool(cma_enable_competition),
                competition_interval=int(cma_competition_interval),
                survival_ratio=float(cma_survival_ratio),
                min_surviving_modes=int(cma_min_surviving_modes),
                elite_ratio=float(cma_elite_ratio),
                rerank_per_mode_after_stage3=bool(rerank_per_mode_after_stage3),
            ),
            stage4=Stage4GradConfig(
                enable=bool(stage4_enable),
                input_stage=stage4_input_stage,
                topk_input=int(stage4_topk_input),
                score_mode=stage4_prob_mode,
                opt_space=stage4_opt_space,
                n_steps=int(stage4_n_steps),
                lr_xy=float(stage4_lr_xy),
                lr_rot=float(stage4_lr_rot),
                lr_scale=float(stage4_lr_scale),
                optimize_scale=bool(stage4_optimize_scale),
                verbose=bool(stage4_verbose),
                log_interval=int(stage4_log_interval),
            ),
            metadata={
                "verbose_stage_timing": bool(debug_stage_timing),
                "stage1_survive_stand": stage1_survive_stand,
                "stage1_move_stand": stage1_move_stand,
            },
        )
        pipeline = SeedModeSearchPipeline(
            seed_screening=TopNSeedScreening(),
            seed_cloud_builder=LocalSeedCloudBuilder(),
            seed_cloud_relocator=BatchedMultiStartEvolutionSeedCloudRefiner(),
            mode_deduper=PassthroughModeDeduper(),
            final_mode_optimizer=EvoTorchFinalModeOptimizer() if bool(stage3_enable) else None,
            stage4_optimizer=GradientTopKOptimizer() if bool(stage4_enable) else None,
            config=pipeline_cfg,
        )

        num_queries = int(coords_gt_all.shape[0])
        seed_coords_all = []
        seed_scores_all = []
        mode_init_all = []
        mode_final_all = []
        query_results = []
        t_start = time.perf_counter()
        mode_opt_elapsed_s = 0.0
        mode_plus_cma_elapsed_s = 0.0
        cma_elapsed_s = 0.0
        query_iter = range(num_queries)
        query_progress = None
        if not bool(debug_stage_timing):
            query_progress = tqdm.tqdm(
                query_iter,
                total=num_queries,
                desc="SeedMode queries",
                leave=True,
            )
            query_iter = query_progress
        try:
            for q_idx in query_iter:
                t_query0 = time.perf_counter()
                l0_prob_q = l0_prob_4d_all[q_idx]
                t_seed_build0 = time.perf_counter()
                seed_coords, seed_scores = self._build_cma_init_seeds(
                    l0_prob_q=l0_prob_q,
                    coords_candidates_flat=coords_candidates_flat,
                    n_coarse=n_coarse,
                    topk_seed=topk_seed,
                    scale_select_mode=scale_select_mode,
                    selection_space=l0_seed_selection_space,
                )
                seed_build_s = time.perf_counter() - t_seed_build0
                seed_coords_all.append(seed_coords)
                seed_scores_all.append(seed_scores)
                feat_q = feats_vis_all[q_idx:q_idx + 1]

                def _make_score_coords_fn(prob_mode):
                    def _score_coords(coords_raw_nx4):
                        with torch.no_grad():
                            scores = evaluator.score_candidates(
                                query_feats=feat_q,
                                coords_batch=coords_raw_nx4.unsqueeze(0),
                                mode=prob_mode,
                                normalize=False,
                            )
                        return torch.nan_to_num(scores.reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
                    return _score_coords

                query_context = {
                    "query_feat": feat_q,
                    "coords_processor": self.coord_normer,
                    "stage1_bin_sizes_raw": stage1_bin_sizes_raw,
                    "stage1_score_chunk_size": int(chunk_size)*2,
                    "stage4_score_chunk_size": int(chunk_size),
                    "score_coords_fn_stage1": _make_score_coords_fn(stage1_prob_mode),
                    "score_coords_fn": _make_score_coords_fn(stage1_prob_mode),
                    "score_coords_fn_stage4": _make_score_coords_fn(stage4_prob_mode),
                    "score_pair_fn": _score_pair_fn_for_refiner,
                    "cma_refiner": cma_refiner,
                    "grid_module": self.grid,
                    "grid_mlp_module": self.grid_mlp,
                    "pos_encoder_grid_module": self.pos_encoder_grid,
                    "get_feats_fm_grid_fn": self._get_feats_fm_grid,
                }
                query_result = pipeline.run_query(
                    query_id=q_idx,
                    coarse_seed_coords_raw=seed_coords,
                    coarse_seed_scores=seed_scores,
                    query_context=query_context,
                )
                query_timing = dict(query_result.metadata.get("timing", {}) or {})
                query_timing["seed_build_s"] = float(seed_build_s)
                query_result.metadata["timing"] = query_timing
                mode_opt_elapsed_s += float(seed_build_s) + float(query_timing.get("mode_opt_s", 0.0))
                mode_plus_cma_elapsed_s += float(seed_build_s) + float(query_timing.get("mode_plus_stage3_s", query_timing.get("query_total_s", 0.0)))
                cma_elapsed_s += float(query_timing.get("stage3_cma_s", 0.0))
                if bool(debug_stage_timing):
                    print(
                        f"[SeedMode] Query{q_idx} pipeline returned | "
                        f"n_init={len(query_result.modes_init)} "
                        f"n_before_stage3={len(query_result.modes_before_stage3)} "
                        f"n_final={len(query_result.modes_final)} | "
                        f"{time.perf_counter() - t_query0:.3f}s"
                    )
                if len(query_result.modes_init) == 0:
                    fallback_coord = seed_coords[:1]
                    fallback_score = seed_scores[:1]
                    query_result.modes_init = pipeline.seed_cloud_relocator.relocate_seed_clouds(
                        seed_clouds=[],
                        config=pipeline_cfg,
                        query_context=query_context,
                    )
                    query_result.modes_final = []
                    from trainers.util_stage3_multi_stage_refiner import ModeState, QueryStageTrace

                    fallback_mode = ModeState(
                        query_id=q_idx,
                        mode_id=f"q{q_idx}_fallback",
                        center_raw=fallback_coord[0].clone(),
                        sigma_diag_raw=stage1_bin_sizes_raw.clone() / 2.0,
                        score_mode=cma_prob_mode,
                        best_coord_raw=fallback_coord[0].clone(),
                        best_score=float(fallback_score[0].item()),
                        latest_metric=float(fallback_score[0].item()),
                    )
                    query_result.modes_init = [fallback_mode]
                    query_result.modes_before_stage3 = [fallback_mode]
                    query_result.modes_final = [fallback_mode]
                    query_result.stage_trace = QueryStageTrace(query_id=int(q_idx))
                    query_result.stage_trace.add_record(
                        pipeline._make_stage_record(
                            query_id=q_idx,
                            stage_id=15,
                            stage_name="stage1.5",
                            score_func_name=stage1_prob_mode,
                            modes=query_result.modes_init,
                        )
                    )
                    query_result.stage_trace.add_record(
                        pipeline._make_stage_record(
                            query_id=q_idx,
                            stage_id=30,
                            stage_name="stage3",
                            score_func_name=cma_prob_mode,
                            modes=query_result.modes_final,
                        )
                    )
                if len(query_result.modes_final) == 0:
                    query_result.modes_final = [m.clone() for m in query_result.modes_before_stage3]
                    if len(query_result.modes_final) == 0:
                        query_result.modes_final = [m.clone() for m in query_result.modes_init]

                mode_init_all.append(query_result.modes_init)
                mode_final_all.append(query_result.modes_final)
                query_results.append(query_result)
        finally:
            if query_progress is not None:
                query_progress.close()

        coords_evo_fallback, scores_evo_fallback = self._pack_mode_states_for_eval(mode_final_all)
        coords_grid_all = torch.stack(seed_coords_all, dim=0)
        scores_grid_all = torch.stack(seed_scores_all, dim=0)
        coords_mode_all, scores_mode_all = self._pack_mode_states_for_eval(mode_init_all)
        coords_evo_all, scores_evo_all = self._pack_stage_records_for_eval(
            query_results=query_results,
            stage_name="stage3",
            fallback_coords=coords_evo_fallback,
            fallback_scores=scores_evo_fallback,
        )
        elapsed_s = time.perf_counter() - t_start

        coarse_retrieval_report = evaluator.evaluate_and_report(
            coords_grid_all,
            coords_gt_all,
            tag="Coarse-Retrieval",
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
            return_details=True,
        )
        seed_mode_init_report = evaluator.evaluate_and_report(
            coords_mode_all,
            coords_gt_all,
            tag="Seed-Mode-Init",
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
            return_details=True,
        )
        seed_mode_final_report = evaluator.evaluate_and_report(
            coords_evo_all,
            coords_gt_all,
            tag="Seed-Mode-Final",
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
            return_details=True,
        )
        progressive_recall_delta_report = _build_progressive_recall_delta_report(
            {
                "coarse_retrieval": coarse_retrieval_report,
                "seed_mode_init": seed_mode_init_report,
                "seed_mode_final": seed_mode_final_report,
            }
        )
        elapsed_ms = elapsed_s * 1000.0
        per_sample_ms = elapsed_ms / max(num_queries, 1)
        mode_opt_ms = mode_opt_elapsed_s * 1000.0
        mode_opt_per_sample_ms = mode_opt_ms / max(num_queries, 1)
        mode_plus_cma_ms = mode_plus_cma_elapsed_s * 1000.0
        mode_plus_cma_per_sample_ms = mode_plus_cma_ms / max(num_queries, 1)
        cma_ms = cma_elapsed_s * 1000.0
        cma_per_sample_ms = cma_ms / max(num_queries, 1)
        print(
            f"[Seed-Mode Timing] samples={num_queries}, "
            f"mode_opt_total={mode_opt_ms:.2f}ms, mode_opt_per_sample={mode_opt_per_sample_ms:.2f}ms, "
            f"mode_plus_cma_total={mode_plus_cma_ms:.2f}ms, mode_plus_cma_per_sample={mode_plus_cma_per_sample_ms:.2f}ms, "
            f"cma_only_total={cma_ms:.2f}ms, cma_only_per_sample={cma_per_sample_ms:.2f}ms"
        )

        dist_th_nrc = evaluator.final_eval_cfg.get("dist_th", None)
        if dist_th_nrc is None:
            dist_th_nrc = float(evaluator.trainer.sat_dataset.halfimg_radius_nrc) * float(evaluator.final_eval_cfg["dist_lambda"])
        else:
            dist_th_nrc = float(dist_th_nrc)
        nrc2meter = None
        dist_th_m = None
        if hasattr(evaluator.trainer.sat_dataset, "halfimg_radius_meter") and hasattr(evaluator.trainer.sat_dataset, "halfimg_radius_nrc"):
            nrc2meter = float(evaluator.trainer.sat_dataset.halfimg_radius_meter) / max(
                float(evaluator.trainer.sat_dataset.halfimg_radius_nrc), 1e-8
            )
            dist_th_m = dist_th_nrc * nrc2meter

        seed_mode_eval_config = {
            "source_function": "_test_3d_fine_accuracy_seed_mode_CMA_ES",
            "n_samples": None if n_samples is None else int(n_samples),
            "use_train_uav": bool(use_train_uav),
            "temperature": float(temperature),
            "shuffle": bool(shuffle),
            "save_pred_pdf": bool(save_pred_pdf),
            "enable_filter": bool(enable_filter),
            "chunk_size": int(chunk_size),
            "n_bins_4d_requested": None if n_bins_4d is None else [int(x) for x in n_bins_4d],
            "n_bins_scale_mode": str(n_bins_scale_mode),
            "l0_prob_mode": str(l0_prob_mode),
            "l0_topN": None if l0_topN is None else int(l0_topN),
            "l0_ratio": None if l0_ratio is None else float(l0_ratio),
            "stage1_mode_input_max": None if stage1_mode_input_max is None else int(stage1_mode_input_max),
            "l0_topk_seed_resolved": int(topk_seed),
            "l0_seed_resolution": dict(l0_seed_resolution),
            "eval_thresh_cfg": None if eval_thresh_cfg is None else dict(eval_thresh_cfg),
            "eval_thresh_cfg_resolved": {
                "dist_lambda": float(evaluator.final_eval_cfg["dist_lambda"]),
                "dist_th": float(dist_th_nrc),
                "dist_th_m": None if dist_th_m is None else float(dist_th_m),
                "nrc2meter": None if nrc2meter is None else float(nrc2meter),
                "rot_th": None if evaluator.final_eval_cfg["rot_th"] is None else float(evaluator.final_eval_cfg["rot_th"]),
                "scale_ratio_th": None if evaluator.final_eval_cfg["scale_ratio_th"] is None else float(evaluator.final_eval_cfg["scale_ratio_th"]),
            },
            "scale_select_mode": str(scale_select_mode),
            "l0_seed_selection_space": str(l0_seed_selection_space),
            "stage1_alpha": float(stage1_alpha),
            "stage1_prob_mode": str(stage1_prob_mode),
            "stage1_samples_per_round": [int(x) for x in self._ensure_param_sequence(stage1_samples_per_round, int)],
            "stage1_radius_scale_per_round": [float(x) for x in self._ensure_param_sequence(stage1_radius_scale_per_round, float)],
            "stage1_local_sample_method": str(stage1_local_sample_method),
            "stage1_elite_ratio": float(stage1_elite_ratio),
            "stage1_survival_ratio_per_round": [float(x) for x in self._ensure_param_sequence(stage1_survival_ratio_per_round, float)],
            "stage1_min_surviving_clouds": int(stage1_min_surviving_clouds),
            "stage1_enable_scale_sampling": bool(stage1_enable_scale_sampling),
            "stage1_5_refiner_mode": str(stage1_5_refiner_mode),
            "stage1_survive_stand": str(stage1_survive_stand),
            "stage1_move_stand": str(stage1_move_stand),
            "cma_prob_mode": str(cma_prob_mode),
            "cma_optimize_scale": bool(cma_optimize_scale),
            "cma_variant": str(cma_variant),
            "cma_init_sigma_manual": None if cma_init_sigma_manual is None else float(cma_init_sigma_manual),
            "cma_popsize": int(cma_popsize),
            "cma_iters": int(cma_iters),
            "cma_enable_early_stop": bool(cma_enable_early_stop),
            "cma_early_stop_patience": int(cma_early_stop_patience),
            "cma_enable_competition": bool(cma_enable_competition),
            "cma_competition_interval": int(cma_competition_interval),
            "cma_survival_ratio": float(cma_survival_ratio),
            "cma_min_surviving_modes": int(cma_min_surviving_modes),
            "cma_elite_ratio": float(cma_elite_ratio),
            "cma_max_input_mode": int(cma_max_input_mode),
            "rerank_per_mode_after_stage3": bool(rerank_per_mode_after_stage3),
            "stage3_enable": bool(stage3_enable),
            "stage4_enable": bool(stage4_enable),
            "stage4_prob_mode": str(stage4_prob_mode),
            "stage4_input_stage": str(stage4_input_stage),
            "stage4_topk_input": int(stage4_topk_input),
            "stage4_opt_space": str(stage4_opt_space),
            "stage4_n_steps": int(stage4_n_steps),
            "stage4_lr_xy": float(stage4_lr_xy),
            "stage4_lr_rot": float(stage4_lr_rot),
            "stage4_lr_scale": float(stage4_lr_scale),
            "stage4_optimize_scale": bool(stage4_optimize_scale),
            "stage4_verbose": bool(stage4_verbose),
            "stage4_log_interval": int(stage4_log_interval),
            "debug_stage_timing": bool(debug_stage_timing),
            "refine_loc": bool(refine_loc),
            "n_coarse_effective": [int(x) for x in n_coarse],
            "n_coarse_3d_effective": [int(x) for x in n_coarse_3d],
            "final_eval_cfg_effective": dict(evaluator.final_eval_cfg),
        }
        seed_mode_reports = {
            "spatial_classification": spatial_classification_report,
            "filtered_spatial_classification": filtered_spatial_classification_report,
            "coarse_retrieval": coarse_retrieval_report,
            "seed_mode_init": seed_mode_init_report,
            "seed_mode_final": seed_mode_final_report,
            "progressive_recall_delta": progressive_recall_delta_report,
            "timing": {
                "samples": int(num_queries),
                "total_s": float(elapsed_s),
                "per_sample_s": float(elapsed_s / max(num_queries, 1)),
                "total_ms": float(elapsed_ms),
                "per_sample_ms": float(per_sample_ms),
                "mode_opt_total_s": float(mode_opt_elapsed_s),
                "mode_opt_per_sample_s": float(mode_opt_elapsed_s / max(num_queries, 1)),
                "mode_opt_total_ms": float(mode_opt_ms),
                "mode_opt_per_sample_ms": float(mode_opt_per_sample_ms),
                "mode_plus_cma_total_s": float(mode_plus_cma_elapsed_s),
                "mode_plus_cma_per_sample_s": float(mode_plus_cma_elapsed_s / max(num_queries, 1)),
                "mode_plus_cma_total_ms": float(mode_plus_cma_ms),
                "mode_plus_cma_per_sample_ms": float(mode_plus_cma_per_sample_ms),
                "cma_only_total_s": float(cma_elapsed_s),
                "cma_only_per_sample_s": float(cma_elapsed_s / max(num_queries, 1)),
                "cma_only_total_ms": float(cma_ms),
                "cma_only_per_sample_ms": float(cma_per_sample_ms),
            },
        }

        return {
            "scores_grid": scores_grid_all,
            "coords_grid": coords_grid_all,
            "scores_mode": scores_mode_all,
            "coords_mode": coords_mode_all,
            "scores_evo": scores_evo_all,
            "coords_evo": coords_evo_all,
            "coords_gt": coords_gt_all,
            "seed_mode_eval_config": seed_mode_eval_config,
            "seed_mode_reports": seed_mode_reports,
            "query_results": query_results,
        }

def _sample_indices(self, scores, n_samples, strategy='sampling', temperature=1.0, alpha=0.8):
        """
        辅助函数：根据分数选择索引
        :param scores: [B, N] 原始分数或概率
        :param n_samples: 需要选择的数量
        :param strategy: 'topk' (原有逻辑), 'sampling' (纯概率采样), 'hybrid' (混合策略)
        :param temperature: 温度系数，越小越趋近于argmax，越大越均匀
        :param alpha: 混合策略中，保留 TopK 的比例 (0.0 - 1.0)
        :return: sampled_indices [B, n_samples], sampled_scores [B, n_samples]
        """
        B, N = scores.shape
        n_samples = min(n_samples, N)

        # 预处理：如果是logits则softmax，如果是概率则归一化
        # 这里假设输入是正值概率或相似度，先进行归一化
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        # 应用温度系数 (注意：如果原本已经是概率，需要先取log再除温度再softmax，或者直接幂次)
        # 这里采用幂次调节法调节由于温度带来的分布平滑度
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if strategy == 'topk':
            return torch.topk(probs, k=n_samples, dim=-1, largest=True)

        elif strategy == 'sampling':
            # 纯概率采样 (无放回)
            indices = torch.multinomial(probs, num_samples=n_samples, replacement=False)
            # 为了后续处理方便，通常还是把采样出来的点按概率从大到小排个序
            batch_indices = torch.arange(B, device=scores.device).unsqueeze(-1)
            selected_probs = scores[batch_indices, indices]
            sorted_idx = torch.argsort(selected_probs, dim=-1, descending=True)
            final_indices = torch.gather(indices, 1, sorted_idx)
            final_scores = torch.gather(selected_probs, 1, sorted_idx)
            return final_scores, final_indices

        elif strategy == 'hybrid':
            # 混合策略：Top (alpha * N) 确定性保留 + 剩余部分随机采样
            # 这种方法既能保住最可能的点（保Top1），又能探索长尾（保Top64）
            n_deterministic = int(n_samples * alpha)
            n_stochastic = n_samples - n_deterministic

            # 1. 先取 Top K
            top_vals, top_inds = torch.topk(probs, k=n_deterministic, dim=-1, largest=True)

            # 2. 将 Top K 的概率置 0，防止重复采样
            probs_clone = probs.clone()
            probs_clone.scatter_(1, top_inds, 0)
            probs_clone = probs_clone / (probs_clone.sum(dim=-1, keepdim=True) + 1e-6)  # 重新归一化

            # 3. 对剩余部分采样
            if n_stochastic > 0:
                stochastic_inds = torch.multinomial(probs_clone, num_samples=n_stochastic, replacement=False)
                final_indices = torch.cat([top_inds, stochastic_inds], dim=1)
            else:
                final_indices = top_inds

            # 同样按原始分数排序输出
            batch_indices = torch.arange(B, device=scores.device).unsqueeze(-1)
            final_raw_scores = scores[batch_indices, final_indices]
            sorted_idx = torch.argsort(final_raw_scores, dim=-1, descending=True)

            final_indices = torch.gather(final_indices, 1, sorted_idx)
            final_scores = torch.gather(final_raw_scores, 1, sorted_idx)

            return final_scores, final_indices

def _test_3d_fine_accuracy_coarse2fine(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=True,
            enable_filter=True,
            chunk_size=2048,
            n_bins_4d=None,
            n_bins_scale_mode="linear",
            lk_prob_mode='ingp',
            l0_prob_mode='ingp',
            l0_topN=512,
            level_resample_cfgs=None,
            eval_thresh_cfg=None,
            refine_loc=True,
            scale_select_mode='argmax',
    ):
        def _execute_hybrid_sampling(coords_candidates, scores, target_topN,
                                     alpha=0.5, threshold=0.0, tag="Level-X"):
            B, N = scores.shape
            K = min(target_topN, N)
            if threshold > 0:
                mask = scores > threshold
                scores = scores * mask.float()
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
            n_det = int(K * alpha)
            n_sto = K - n_det
            _, idx_det = torch.topk(probs, k=n_det, dim=-1)
            if n_sto > 0:
                probs_remain = probs.clone()
                probs_remain.scatter_(1, idx_det, 0.0)
                probs_remain = probs_remain / (probs_remain.sum(dim=-1, keepdim=True) + 1e-10)
                idx_sto = torch.multinomial(probs_remain, num_samples=n_sto, replacement=True)
                final_indices = torch.cat([idx_det, idx_sto], dim=1)
            else:
                final_indices = idx_det
            idx_exp = final_indices.unsqueeze(-1).expand(-1, -1, 4)
            return torch.gather(coords_candidates, 1, idx_exp)

        evaluator = Stage3FineLocManager(
            self,
            temperature=temperature,
            chunk_size=chunk_size,
            eval_thresh_cfg=eval_thresh_cfg,
        )
        lk_prob_mode = evaluator.validate_prob_mode("lk_prob_mode", lk_prob_mode)
        l0_prob_mode = evaluator.validate_prob_mode("l0_prob_mode", l0_prob_mode)

        l0_topN = int(l0_topN)
        if l0_topN <= 0:
            raise ValueError("l0_topN must be > 0.")
        scale_select_mode = str(scale_select_mode).strip().lower()
        if scale_select_mode not in ("argmax", "log_expectation"):
            raise ValueError(
                f"scale_select_mode must be 'argmax' or 'log_expectation', got {scale_select_mode}"
            )

        if level_resample_cfgs is None:
            level_resample_cfgs = [
                {"resample_dims": (4, 4, 2, 1), "topN": 128, "space_scale": 1.0},
                {"resample_dims": (5, 5, 2, 1), "topN": 128, "space_scale": 1.0},
                {"resample_dims": (4, 4, 2, 5), "topN": 128, "space_scale": 1.0},
            ]
        if len(level_resample_cfgs) != 3:
            raise ValueError("level_resample_cfgs must have length 3 for L1~L3.")

        cfgs_norm = []
        for idx, cfg in enumerate(level_resample_cfgs):
            if not isinstance(cfg, dict):
                raise ValueError(f"level_resample_cfgs[{idx}] must be a dict.")
            if "resample_dims" not in cfg or "topN" not in cfg:
                raise ValueError(f"level_resample_cfgs[{idx}] must include resample_dims and topN.")
            resample_dims = tuple(int(x) for x in cfg["resample_dims"])
            if len(resample_dims) != 4 or any(d <= 0 for d in resample_dims):
                raise ValueError(f"level_resample_cfgs[{idx}].resample_dims must be 4D positive.")
            topN_keep = int(cfg["topN"])
            if topN_keep <= 0:
                raise ValueError(f"level_resample_cfgs[{idx}].topN must be > 0.")
            space_scale = float(cfg.get("space_scale", 1.0))
            if space_scale <= 0:
                raise ValueError(f"level_resample_cfgs[{idx}].space_scale must be > 0.")
            level_prob_mode = cfg.get("lk_prob_mode", cfg.get("prob_mode", None))
            if level_prob_mode is None:
                level_prob_mode = lk_prob_mode if idx == 0 else "ingp"
            level_prob_mode = str(level_prob_mode).lower()
            if level_prob_mode not in ("ingp", "projector", "product"):
                raise ValueError(
                    f"level_resample_cfgs[{idx}].prob_mode must be one of "
                    f"('ingp','projector','product'), got {level_prob_mode}"
                )
            cfgs_norm.append({
                "resample_dims": resample_dims,
                "topN": topN_keep,
                "space_scale": space_scale,
                "lk_prob_mode": level_prob_mode,
            })
        level_resample_cfgs = cfgs_norm

        print(f"\n{'=' * 60}")
        print(f"3D分类测试 (NR, NC, Rot)")
        print(f"测试样本数: {n_samples if n_samples else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(f"L0 prob mode: {l0_prob_mode}")
        print(f"{'=' * 60}\n")
        coarse = evaluator.run_l0_coarse_retrieval(
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            shuffle=shuffle,
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode=n_bins_scale_mode,
            l0_prob_mode=l0_prob_mode,
            save_pred_pdf=save_pred_pdf,
            save_tag="prefilter",
            report_title="Spatial Classification Results (wo eval scale)",
        )
        pred_pdf_3d_all = coarse["pred_pdf_3d_all"]
        q_label_3d_all = coarse["q_label_3d_all"]
        coords_gt_all = coarse["coords_gt_all"]
        feats_vis_all = coarse["feats_vis_all"]
        l0_prob_4d_all = coarse["l0_prob_4d_all"]
        coords_candidates_flat = coarse["coords_candidates_flat"]
        n_coarse = coarse["n_coarse"]
        n_coarse_3d = coarse["n_coarse_3d"]
        cell_centers_3d = coarse["cell_centers_3d"]

        coords_topk, scores_topk = evaluator.evaluate_pred_pdf_topk(
            pred_pdf_3d_all=pred_pdf_3d_all,
            cell_centers_3d=cell_centers_3d,
            n_coarse_3d=n_coarse_3d,
            coords_gt_all=coords_gt_all,
        )

        if not refine_loc:
            if coords_topk is None:
                coords_topk = coords_gt_all.unsqueeze(1)
                scores_topk = torch.zeros(
                    (coords_gt_all.shape[0], 1),
                    device=coords_gt_all.device,
                    dtype=coords_gt_all.dtype
                )
            return {
                "probs_fm_dist": scores_topk,
                "coords_pred": coords_topk,
                "coords_gt": coords_gt_all,
            }

        pred_pdf_3d_shaped = pred_pdf_3d_all.reshape(-1, *n_coarse_3d)
        if enable_filter:
            preds_filtered = evaluator.apply_histogram_filter(
                pred_pdf_3d_all=pred_pdf_3d_all,
                coords_gt_all=coords_gt_all,
                n_coarse_3d=n_coarse_3d,
            )
            evaluator.maybe_save_pred_pdf(
                save_pred_pdf=save_pred_pdf,
                pred_pdf_3d_all=preds_filtered,
                q_label_3d_all=q_label_3d_all,
                coords_gt_all=coords_gt_all,
                n_coarse_3d=n_coarse_3d,
                cell_centers_3d=cell_centers_3d,
                use_train_uav=use_train_uav,
                tag="postfilter",
            )
            evaluator.report_spatial_classification(
                pred_pdf_3d_all=preds_filtered.reshape(preds_filtered.shape[0], -1),
                q_label_3d_all=q_label_3d_all,
                title="Spatial Classification Results (wo eval scale), Filtered",
            )
        else:
            preds_filtered = pred_pdf_3d_shaped

        enable_smoothing = False
        smooth_sigma = 0.75
        smooth_kernel = 3
        if enable_smoothing:
            from trainer_depends.utils.util_refine_sampling import _apply_gaussian_smoothing_3d
            preds_filtered = _apply_gaussian_smoothing_3d(
                preds_filtered,
                kernel_size=smooth_kernel,
                sigma=smooth_sigma,
            )

        topN_l0 = l0_topN
        prob_thresh = 0.
        sample_mode = 'hybrid'
        hybrid_alpha_l0 = 0.9
        B = preds_filtered.shape[0]
        probs_flat = preds_filtered.reshape(B, -1)
        if prob_thresh > 0:
            mask_valid = probs_flat > prob_thresh
            probs_flat = probs_flat * mask_valid.float()
        probs_sum = probs_flat.sum(dim=-1, keepdim=True) + 1e-8
        probs_norm = probs_flat / probs_sum

        if sample_mode == 'hybrid':
            n_deterministic = int(topN_l0 * hybrid_alpha_l0)
            n_stochastic = topN_l0 - n_deterministic
            _, indices_top = torch.topk(probs_norm, k=n_deterministic, dim=-1)
            probs_remain = probs_norm.clone()
            probs_remain.scatter_(1, indices_top, 0)
            probs_remain = probs_remain / (probs_remain.sum(dim=-1, keepdim=True) + 1e-8)
            indices_sto = torch.multinomial(probs_remain, num_samples=n_stochastic, replacement=True)
            sampled_indices = torch.cat([indices_top, indices_sto], dim=1)
        else:
            sampled_indices = torch.multinomial(probs_norm, num_samples=topN_l0, replacement=True)

        sampled_indices = sampled_indices.to(coords_candidates_flat.device)
        H, W, O = n_coarse_3d  # [NR, NC, Rot]
        idx_rot = sampled_indices % O
        idx_nc = (sampled_indices // O) % W
        idx_nr = (sampled_indices // (W * O))
        n_scale = n_coarse[3]

        probs_by_anchor = l0_prob_4d_all.reshape(B, H * W * O, n_scale)
        anchor_probs = torch.gather(
            probs_by_anchor,
            1,
            sampled_indices.unsqueeze(-1).expand(-1, -1, n_scale),
        )
        coords_by_anchor = coords_candidates_flat.reshape(H * W * O, n_scale, 4)
        coords_selected = coords_by_anchor.index_select(0, sampled_indices.reshape(-1)).reshape(B, -1, n_scale, 4)
        if scale_select_mode == "argmax":
            best_scale_idx = torch.argmax(anchor_probs, dim=-1)
            row_idx = torch.arange(B, device=coords_candidates_flat.device).unsqueeze(1)
            col_idx = torch.arange(sampled_indices.shape[1], device=coords_candidates_flat.device).unsqueeze(0)
            coords_topN_l0_lefted = coords_selected[row_idx, col_idx, best_scale_idx]
        else:
            weights = anchor_probs / (anchor_probs.sum(dim=-1, keepdim=True) + 1e-8)
            scale_vals = coords_selected[..., 3].clamp(min=1e-6)
            scale_log_hat = torch.sum(weights * torch.log(scale_vals), dim=-1)
            coords_topN_l0_lefted = coords_selected[:, :, 0, :].clone()
            coords_topN_l0_lefted[:, :, 3] = torch.exp(scale_log_hat)

        coords_level = coords_topN_l0_lefted
        prob_lvl = None
        for level_idx, cfg in enumerate(level_resample_cfgs):
            resample_dims = cfg["resample_dims"]
            topN_keep = cfg["topN"]
            space_scale = cfg["space_scale"]

            coords_resampled = self._sample_around_candidates(
                coords_centers=coords_level,  # [B, topN, 4]
                grid_dims=resample_dims,
                space_size=self.subspace_sampler._get_gpu_cache(self.device)['coarse_bin_sizes'] * space_scale,
            )  # -> [B, topN * P, 4]

            coords2eval = torch.cat([coords_resampled, coords_level], dim=1)
            prob_lvl = evaluator.score_candidates(
                query_feats=feats_vis_all,
                coords_batch=coords2eval,
                mode=cfg["lk_prob_mode"],
                normalize=True,
            )
            coords_level = _execute_hybrid_sampling(
                coords_candidates=coords2eval,
                scores=prob_lvl,
                target_topN=topN_keep,
                alpha=1.0,
                threshold=0.00,
                tag=f"Level-{level_idx + 1}"
            )

        coords_resorted_l3 = coords_level

        evaluator.evaluate_and_report(
            coords_resorted_l3,
            coords_gt_all,
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
        )

        dict2ret = {
            "probs_fm_dist":prob_lvl,
            "coords_pred":coords_level,
            "coords_gt":coords_gt_all,
        }
        return dict2ret

def _test_3d_fine_accuracy_CMA_ES(
            self,
            n_samples=256,
            use_train_uav=False,
            temperature=0.5,
            shuffle=False,
            save_pred_pdf=False,
            enable_filter=False,
            chunk_size=2048,
            n_bins_4d=None,
            n_bins_scale_mode="linear",
            l0_prob_mode='ingp',
            l0_topN=64,
            eval_thresh_cfg=None,
            refine_loc=True,
            cma_prob_mode='ingp',
            cma_variant='CMA',
            cma_enable_early_stop=False,
            cma_early_stop_patience=3,
            cma_sigma0=0.10,
            cma_popsize=32,
            cma_iters=16,
            cma_elite_ratio=0.5,
            enable_seed_dedup=True,
            dedup_radius_nrc=0.05,
            dedup_radius_deg=12.0,
            dedup_scale_rel=0.12,
            scale_select_mode='argmax',
            cma_share_gpu_eval_across_queries=True,
            cma_return_final_population=False,
            cma_query_chunk_size=None,
    ):
        """
        V2: L0 粗筛 + EvoTorch 多起点 CMA-ES 精修 的 3D 细粒度定位评估。

        参数说明（调参建议）:
            n_samples (int|None):
                评估样本数。None 表示全量；先调参建议 64/128 小样本快速迭代。
            use_train_uav (bool):
                True 用训练集评估，False 用测试集评估。
            temperature (float):
                Projector 路径 softmax 温度（仅影响 projector 概率分布锐度）。
                值越大分布越尖锐，建议在 [0.05, 2.0] 范围试验。
            shuffle (bool):
                DataLoader 是否打乱。
            save_pred_pdf (bool):
                是否保存 coarse 3D 概率分布 npz 文件。
            enable_filter (bool):
                是否启用时序 HistogramFilter3D（仅 coarse 评估报告使用）。
            chunk_size (int):
                候选点评分时分块大小，显存不足时减小（如 512/1024）。
            n_bins_4d (list[int]|None):
                自定义 L0 网格分辨率 [NR, NC, Rot, Scale]。
                None 时使用 sampler 默认 coarse 网格。
            l0_prob_mode (str):
                L0 粗筛评分模式: {'ingp','projector','product'}。
            l0_topN (int):
                每个 query 进入 CMA 的初始 seed 数量（计算量近似线性增长）。
            eval_thresh_cfg (dict|None):
                评估阈值覆盖，如 {"dist_lambda":1.1,"rot_th":11.0,"scale_ratio_th":None}。
            refine_loc (bool):
                是否执行 CMA 精修；False 时只输出 L0 seed 排序结果。

            cma_prob_mode (str):
                CMA 迭代时的评分模式: {'ingp','projector','product'}。
                与 l0_prob_mode 可不同，用于对比“粗筛信号”和“精修信号”。
            cma_variant (str):
                EvoTorch CMA 后端方案: {'CMA','Sep-CMA'}。
                - 'CMA': 标准全协方差 CMA-ES
                - 'Sep-CMA': 仅对角协方差，更快更省算力
            cma_enable_early_stop (bool):
                是否启用“连续若干代最佳分数无提升”的早停逻辑。
            cma_early_stop_patience (int):
                当启用早停时，单个 seed 连续多少代“最佳分数无提升”就停止。
            cma_sigma0 (float):
                CMA 初始步长（linear 空间）。过大易发散，过小易局部困住。
                常用起点 0.10~0.30。
            cma_popsize (int):
                每个 seed 每代采样数（种群大小），越大越稳但更慢。
                调参常用: 8 / 12 / 16 / 24。
            cma_iters (int):
                每个 seed 的 CMA 迭代代数，越大越精细但更慢。
                调参常用: 4 / 6 / 8 / 12。
            cma_elite_ratio (float):
                兼容参数位。当前 EvoTorch 实现未直接使用该参数。

            enable_seed_dedup (bool):
                是否对 L0 TopN seed 去重，避免多个 seed 落在同一峰。
            dedup_radius_nrc (float):
                seed 去重的平面距离阈值（nrc 坐标）。
            dedup_radius_deg (float):
                seed 去重的旋转阈值（度）。
            dedup_scale_rel (float):
                seed 去重的尺度相对误差阈值。
            scale_select_mode (str):
                每个 3D anchor 内如何解析 scale:
                - 'argmax': 取响应最高的离散 scale
                - 'log_expectation': 对 log-scale 做加权平均
            cma_share_gpu_eval_across_queries (bool):
                True 时，把所有 query 的独立 CMA 进程候选合并为共享 GPU 批评估。
                每个 seed 仍保持独立分布与独立更新，只共享一次前向评分。
            cma_return_final_population (bool):
                是否额外返回每个 query / seed 最后一代 population 及其分数。
            cma_query_chunk_size (int|None):
                CMA 精修阶段每次并行处理多少个 query。
                设小一些可以显著降低显存峰值；None 表示一次处理全部 query。

        计算复杂度:
            近似与 `n_samples * l0_topN * cma_popsize * cma_iters` 成正比。
            建议先固定较小预算快速扫参，再逐步放大。

        推荐起步配置:
            - 快速调参: l0_topN=32, cma_popsize=8,  cma_iters=4
            - 平衡配置: l0_topN=64, cma_popsize=12, cma_iters=6
            - 高精配置: l0_topN=96, cma_popsize=16, cma_iters=8

        返回:
            dict:
                {
                    "probs_fm_dist": [N, K],  # 每个 query 的 TopK 分数
                    "coords_pred":   [N, K, 4],  # 每个 query 的 TopK 坐标 (nr,nc,rot,scale)
                    "coords_gt":     [N, 4],  # GT 坐标
                }
        """
        evaluator = Stage3FineLocManager(
            self,
            temperature=temperature,
            chunk_size=chunk_size,
            eval_thresh_cfg=eval_thresh_cfg,
        )

        from trainers.util_stage3_multi_start_CMAES_by_evotorch import MultiStartCMAESEvoTorchRefiner
        cma_refiner = MultiStartCMAESEvoTorchRefiner(coords_processor=self.coord_normer, device=self.device)

        def _score_pair_fn_for_refiner(query_feats_mxc, coords_raw_mx4):
            with torch.no_grad():
                scores = evaluator.score_candidates(
                    query_feats=query_feats_mxc,
                    coords_batch=coords_raw_mx4.unsqueeze(1),
                    mode=cma_prob_mode,
                    normalize=False,
                )
            scores = scores.reshape(-1)
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
            return scores

        l0_prob_mode = evaluator.validate_prob_mode("l0_prob_mode", l0_prob_mode)
        cma_prob_mode = evaluator.validate_prob_mode("cma_prob_mode", cma_prob_mode)
        cma_variant_raw = str(cma_variant)
        cma_variant_norm = cma_variant_raw.strip().lower()
        if cma_variant_norm in ("cma", "standard", "base"):
            cma_variant = "CMA"
        elif cma_variant_norm in ("sep-cma", "separable", "sep", "diag"):
            cma_variant = "Sep-CMA"
        else:
            raise ValueError(f"cma_variant must be 'CMA' or 'Sep-CMA', got {cma_variant_raw}")
        cma_enable_early_stop = bool(cma_enable_early_stop)
        cma_early_stop_patience = max(1, int(cma_early_stop_patience))
        if not bool(cma_share_gpu_eval_across_queries):
            print("[V2] EvoTorch refiner currently always shares GPU evaluation across active queries.")
        l0_topN = int(l0_topN)
        if l0_topN <= 0:
            raise ValueError("l0_topN must be > 0")
        scale_select_mode = str(scale_select_mode).strip().lower()
        if scale_select_mode not in ("argmax", "log_expectation"):
            raise ValueError(
                f"scale_select_mode must be 'argmax' or 'log_expectation', got {scale_select_mode}"
            )
        if cma_query_chunk_size is None:
            cma_query_chunk_size = None
        else:
            cma_query_chunk_size = max(1, int(cma_query_chunk_size))

        print(f"\n{'=' * 60}")
        print("3D细粒度定位测试 (EvoTorch CMA-ES Multi-Start)")
        print(f"测试样本数: {n_samples if n_samples is not None else '全部'}")
        print(f"数据集: {'训练集' if use_train_uav else '测试集'}")
        print(
            f"L0 mode: {l0_prob_mode} | CMA mode: {cma_prob_mode} | "
            f"CMA variant: {cma_variant} | early_stop: {cma_enable_early_stop} | "
            f"patience: {cma_early_stop_patience}"
        )
        print(f"Seed init: 3D-anchor + {scale_select_mode}")
        print(f"Shared GPU eval across queries: {bool(cma_share_gpu_eval_across_queries)}")
        print(f"CMA query chunk size: {cma_query_chunk_size if cma_query_chunk_size is not None else 'all'}")
        print(f"{'=' * 60}\n")
        coarse = evaluator.run_l0_coarse_retrieval(
            n_samples=n_samples,
            use_train_uav=use_train_uav,
            shuffle=shuffle,
            n_bins_4d=n_bins_4d,
            n_bins_scale_mode=n_bins_scale_mode,
            l0_prob_mode=l0_prob_mode,
            save_pred_pdf=save_pred_pdf,
            save_tag="prefilter_v2",
            report_title="Spatial Classification Results (V2, wo eval scale)",
        )
        pred_pdf_3d_all = coarse["pred_pdf_3d_all"]
        q_label_3d_all = coarse["q_label_3d_all"]
        coords_gt_all = coarse["coords_gt_all"]
        feats_vis_all = coarse["feats_vis_all"]
        l0_prob_4d_all = coarse["l0_prob_4d_all"]
        coords_candidates_flat = coarse["coords_candidates_flat"]
        n_coarse = coarse["n_coarse"]
        n_coarse_3d = coarse["n_coarse_3d"]

        if enable_filter:
            preds_filtered = evaluator.apply_histogram_filter(
                pred_pdf_3d_all=pred_pdf_3d_all,
                coords_gt_all=coords_gt_all,
                n_coarse_3d=n_coarse_3d,
            )
            evaluator.report_spatial_classification(
                pred_pdf_3d_all=preds_filtered.reshape(preds_filtered.shape[0], -1),
                q_label_3d_all=q_label_3d_all,
                title="Spatial Classification Results, Filtered (V2, wo eval scale)",
            )

        num_queries = int(coords_gt_all.shape[0])
        n_anchor_3d = int(np.prod(n_coarse_3d))
        topk_seed = min(l0_topN, n_anchor_3d)
        init_coords_list = []
        init_scores_list = []
        cma_backend_available = True
        cma_timer_start = time.perf_counter()
        for q_idx in range(num_queries):
            l0_prob_q = l0_prob_4d_all[q_idx]
            seed_coords, seed_scores = self._build_cma_init_seeds(
                l0_prob_q=l0_prob_q,
                coords_candidates_flat=coords_candidates_flat,
                n_coarse=n_coarse,
                topk_seed=topk_seed,
                scale_select_mode=scale_select_mode,
            )
            seed_coords, seed_scores = cma_refiner.select_diverse_seeds(
                coords_sorted=seed_coords,
                scores_sorted=seed_scores,
                target_k=topk_seed,
                enable_seed_dedup=enable_seed_dedup,
                dedup_radius_nrc=dedup_radius_nrc,
                dedup_radius_deg=dedup_radius_deg,
                dedup_scale_rel=dedup_scale_rel,
            )
            init_coords_list.append(seed_coords[:topk_seed])
            init_scores_list.append(seed_scores[:topk_seed])

        coords_init_all = torch.stack(init_coords_list, dim=0)
        probs_init_all = torch.stack(init_scores_list, dim=0)
        del init_coords_list
        del init_scores_list
        del l0_prob_4d_all

        final_population_coords = None
        final_population_scores = None
        query_chunk_size = num_queries if cma_query_chunk_size is None else min(cma_query_chunk_size, num_queries)
        if refine_loc and cma_backend_available:
            try:
                refined_coords_chunks = []
                refined_score_chunks = []
                final_population_coords_chunks = []
                final_population_scores_chunks = []
                for chunk_start in range(0, num_queries, query_chunk_size):
                    chunk_end = min(chunk_start + query_chunk_size, num_queries)
                    refine_result = cma_refiner.refine_batch_queries(
                        query_feats=feats_vis_all[chunk_start:chunk_end],
                        seed_coords_raw_batch=coords_init_all[chunk_start:chunk_end],
                        score_pair_fn=_score_pair_fn_for_refiner,
                        sigma0=cma_sigma0,
                        popsize=cma_popsize,
                        n_iterations=cma_iters,
                        maximize=True,
                        cma_seed=chunk_start,
                        cma_variant=cma_variant,
                        enable_early_stop=cma_enable_early_stop,
                        early_stop_patience=cma_early_stop_patience,
                        return_diagnostics=cma_return_final_population,
                    )
                    if cma_return_final_population:
                        refined_coords_chunks.append(refine_result["best_coords"])
                        refined_score_chunks.append(refine_result["best_scores"])
                        final_population_coords_chunks.append(refine_result["final_population_coords"])
                        final_population_scores_chunks.append(refine_result["final_population_scores"])
                    else:
                        coords_chunk, scores_chunk = refine_result
                        refined_coords_chunks.append(coords_chunk)
                        refined_score_chunks.append(scores_chunk)

                coords_refined_all = torch.cat(refined_coords_chunks, dim=0)
                score_refined_all = torch.cat(refined_score_chunks, dim=0)
                if cma_return_final_population:
                    final_population_coords = torch.cat(final_population_coords_chunks, dim=0)
                    final_population_scores = torch.cat(final_population_scores_chunks, dim=0)
            except ImportError as err:
                cma_backend_available = False
                print(f"[V2] EvoTorch CMA backend unavailable ({err}). Fallback to seed ranking.")
                coords_refined_all = coords_init_all
                score_refined_chunks = []
                with torch.no_grad():
                    for chunk_start in range(0, num_queries, query_chunk_size):
                        chunk_end = min(chunk_start + query_chunk_size, num_queries)
                        scores_chunk = evaluator.score_candidates(
                            query_feats=feats_vis_all[chunk_start:chunk_end],
                            coords_batch=coords_refined_all[chunk_start:chunk_end],
                            mode=cma_prob_mode,
                            normalize=False,
                        )
                        scores_chunk = torch.nan_to_num(scores_chunk, nan=-1e9, posinf=1e9, neginf=-1e9)
                        score_refined_chunks.append(scores_chunk)
                score_refined_all = torch.cat(score_refined_chunks, dim=0)
        else:
            coords_refined_all = coords_init_all
            score_refined_chunks = []
            with torch.no_grad():
                for chunk_start in range(0, num_queries, query_chunk_size):
                    chunk_end = min(chunk_start + query_chunk_size, num_queries)
                    scores_chunk = evaluator.score_candidates(
                        query_feats=feats_vis_all[chunk_start:chunk_end],
                        coords_batch=coords_refined_all[chunk_start:chunk_end],
                        mode=cma_prob_mode,
                        normalize=False,
                    )
                    scores_chunk = torch.nan_to_num(scores_chunk, nan=-1e9, posinf=1e9, neginf=-1e9)
                    score_refined_chunks.append(scores_chunk)
            score_refined_all = torch.cat(score_refined_chunks, dim=0)

        rank_idx = torch.argsort(score_refined_all, dim=1, descending=True)
        coords_pred_all = torch.gather(
            coords_refined_all,
            dim=1,
            index=rank_idx.unsqueeze(-1).expand(-1, -1, coords_refined_all.shape[-1]),
        )[:, :topk_seed]
        probs_pred_all = torch.gather(
            score_refined_all,
            dim=1,
            index=rank_idx,
        )[:, :topk_seed]

        cma_elapsed_s = time.perf_counter() - cma_timer_start
        evaluator.evaluate_and_report(
            coords_init_all,
            coords_gt_all,
            tag="CMA-ES-InitSeeds-V2",
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
        )
        evaluator.evaluate_and_report(
            coords_pred_all,
            coords_gt_all,
            tag="CMA-ES-Refined-V2",
            dist_lambda=evaluator.final_eval_cfg["dist_lambda"],
            rot_th=evaluator.final_eval_cfg["rot_th"],
            scale_ratio_th=evaluator.final_eval_cfg["scale_ratio_th"],
            scale_select_mode=scale_select_mode,
        )
        avg_cma_elapsed_s = cma_elapsed_s / max(num_queries, 1)
        print(
            f"[CMA-ES Timing] samples={num_queries}, total={cma_elapsed_s:.3f}s, "
            f"per_sample={avg_cma_elapsed_s:.3f}s"
        )

        dict2ret = {
            "probs_fm_dist": probs_pred_all,
            "coords_pred": coords_pred_all,
            "coords_gt": coords_gt_all,
        }
        if cma_return_final_population:
            dict2ret["final_population_coords"] = final_population_coords
            dict2ret["final_population_scores"] = final_population_scores
        return dict2ret
