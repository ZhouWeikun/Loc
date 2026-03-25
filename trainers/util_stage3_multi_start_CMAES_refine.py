import numpy as np
import torch

try:
    from cmaes import CMA
    _CMAES_AVAILABLE = True
except Exception:
    CMA = None
    _CMAES_AVAILABLE = False


class MultiStartCMAESRefiner:
    """
    Multi-start CMA-ES refiner for 4D localization coordinates.

    Coordinate convention:
        raw   : [nr, nc, theta_rad, scale]
        linear: [nr_n, nc_n, theta_lin, scale_n], each nominally in [-1, 1]
    """

    def __init__(self, coords_processor, device=None):
        """
        Args:
            coords_processor: instance with `raw_to_linear` and `linear_to_raw`
            device: torch device. If None, uses coords_processor.device
        """
        self.processor = coords_processor
        self.device = device if device is not None else coords_processor.device

    # ------------------------------------------------------------------
    # Basic coordinate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_theta_linear(theta_lin):
        """
        Wrap linear-theta from R to [-1, 1).
        """
        return torch.remainder(theta_lin + 1.0, 2.0) - 1.0

    def project_linear_bounds(self, coords_linear):
        """
        Project linear coordinates to valid ranges.
        """
        coords = coords_linear.clone()
        coords[..., 0:2] = coords[..., 0:2].clamp(-1.0, 1.0)
        coords[..., 2:3] = self._wrap_theta_linear(coords[..., 2:3])
        coords[..., 3:4] = coords[..., 3:4].clamp(-1.0, 1.0)
        return coords

    @staticmethod
    def cyclic_rot_diff_deg(theta_a, theta_b):
        """
        Cyclic rotation difference in degree for raw theta (radian).
        """
        diff = torch.abs(theta_a - theta_b)
        diff = torch.minimum(diff, 2 * torch.pi - diff)
        return diff * 180.0 / torch.pi

    # ------------------------------------------------------------------
    # Seed selection helper
    # ------------------------------------------------------------------
    def select_diverse_seeds(
        self,
        coords_sorted,
        scores_sorted,
        target_k,
        enable_seed_dedup=True,
        dedup_radius_nrc=0.08,
        dedup_radius_deg=12.0,
        dedup_scale_rel=0.12,
    ):
        """
        Select top-k seeds with optional geometric deduplication.

        Args:
            coords_sorted: [N, 4] raw coords sorted by descending score
            scores_sorted: [N] corresponding scores
            target_k: number of seeds to keep
            enable_seed_dedup: whether to enforce diversity constraints
        """
        coords_sorted = self._to_tensor(coords_sorted)
        scores_sorted = self._to_tensor(scores_sorted).reshape(-1)
        target_k = max(1, int(target_k))

        if (not enable_seed_dedup) or coords_sorted.shape[0] <= 1:
            keep_k = min(target_k, coords_sorted.shape[0])
            return coords_sorted[:keep_k], scores_sorted[:keep_k]

        selected = []
        total = int(coords_sorted.shape[0])
        for idx in range(total):
            if len(selected) >= target_k:
                break
            cand = coords_sorted[idx]
            keep = True
            for prev_idx in selected:
                prev = coords_sorted[prev_idx]
                dist_nrc = torch.norm(cand[:2] - prev[:2], p=2).item()
                diff_rot_deg = self.cyclic_rot_diff_deg(cand[2], prev[2]).item()
                denom = max(float(torch.maximum(cand[3], prev[3]).item()), 1e-6)
                diff_scale_rel = float(torch.abs(cand[3] - prev[3]).item()) / denom
                if (
                    dist_nrc <= dedup_radius_nrc
                    and diff_rot_deg <= dedup_radius_deg
                    and diff_scale_rel <= dedup_scale_rel
                ):
                    keep = False
                    break
            if keep:
                selected.append(idx)

        # Back-fill if dedup removes too many points.
        if len(selected) < target_k:
            for idx in range(total):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_k:
                    break

        if len(selected) == 0:
            selected = [0]
        id_keep = torch.tensor(selected[:target_k], dtype=torch.long, device=coords_sorted.device)
        return coords_sorted[id_keep], scores_sorted[id_keep]

    # ------------------------------------------------------------------
    # Main optimization API
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_cma_variant(cma_variant):
        variant = str(cma_variant).strip().lower()
        if variant in ("cma", "standard", "base"):
            return "CMA"
        if variant in ("lra-cma", "lracma", "lra_cma", "lra"):
            return "LRA-CMA"
        raise ValueError(f"Unsupported cma_variant: {cma_variant}. Use 'CMA' or 'LRA-CMA'.")

    @staticmethod
    def _optimizer_should_stop(opt):
        should_stop = getattr(opt, "should_stop", None)
        if not callable(should_stop):
            return False
        try:
            return bool(should_stop())
        except Exception:
            return False

    def refine_single_query(
        self,
        query_feat,
        seed_coords_raw,
        score_fn,
        sigma0=0.20,
        popsize=16,
        n_iterations=8,
        elite_ratio=None,
        maximize=True,
        cma_seed=None,
        cma_variant="CMA",
        enable_early_stop=False,
        early_stop_patience=3,
    ):
        """
        Run multi-start CMA-ES for one query feature.

        Args:
            query_feat: [1, C] or [C]
            seed_coords_raw: [K, 4], K starts
            score_fn: callable(query_feat_1xc, coords_raw_nx4) -> [N] or [1, N]
            sigma0: initial std in linear space
            popsize: population size per start
            n_iterations: optimization generations
            elite_ratio: reserved compatibility argument.
                Note: standard `cmaes.CMA` does not expose elite-ratio as a direct knob.
            maximize: True if score_fn is "higher is better"
            cma_seed: base random seed for CMA instances
            cma_variant: 'CMA' or 'LRA-CMA'
                - 'CMA': 标准 CMA-ES
                - 'LRA-CMA': 同一后端下启用 learning-rate adaptation (lr_adapt=True)
            enable_early_stop: 是否启用后端 stop criteria 早停（需要 cmaes.CMA.should_stop）
            early_stop_patience: 连续多少代“最佳分数无提升”后停止该 start

        Returns:
            best_raw: [K, 4] best coord per start
            best_scores: [K] best score per start
        """
        self._ensure_cmaes_available()

        query_feat_t = self._to_tensor(query_feat)
        if query_feat_t.ndim == 1:
            query_feat_t = query_feat_t.unsqueeze(0)

        seeds_raw_t = self._to_tensor(seed_coords_raw)
        if seeds_raw_t.ndim != 2 or seeds_raw_t.shape[1] != 4:
            raise ValueError("seed_coords_raw must be [K, 4]")

        n_start = int(seeds_raw_t.shape[0])
        if n_start == 0:
            empty_coords = torch.zeros((0, 4), device=self.device, dtype=torch.float32)
            empty_scores = torch.zeros((0,), device=self.device, dtype=torch.float32)
            return empty_coords, empty_scores

        popsize = max(4, int(popsize))
        n_iterations = max(1, int(n_iterations))
        sigma0 = float(sigma0)
        sigma0 = max(1e-4, sigma0)
        cma_variant = self._normalize_cma_variant(cma_variant)
        use_lra = cma_variant == "LRA-CMA"
        enable_early_stop = bool(enable_early_stop)
        early_stop_patience = max(1, int(early_stop_patience))

        # Convert starts to linear means.
        means_linear_t = self.processor.raw_to_linear(seeds_raw_t)
        means_linear_t = self.project_linear_bounds(means_linear_t)
        means_linear_np = means_linear_t.detach().cpu().numpy()

        # Fixed bounds in linear space.
        bounds = np.array(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )

        optimizers = []
        for i in range(n_start):
            seed_i = None if cma_seed is None else int(cma_seed) + i
            cma_kwargs = dict(
                mean=means_linear_np[i].astype(np.float64),
                sigma=sigma0,
                bounds=bounds,
                population_size=popsize,
                seed=seed_i,
            )
            if use_lra:
                cma_kwargs["lr_adapt"] = True
            try:
                opt = CMA(**cma_kwargs)
            except TypeError as err:
                if use_lra and "lr_adapt" in str(err):
                    raise RuntimeError(
                        "Current `cmaes` version does not support LRA-CMA (`lr_adapt`). "
                        "Please upgrade `cmaes` or switch cma_variant='CMA'."
                    ) from err
                raise
            optimizers.append(opt)

        # Initialize best with seed scores.
        with torch.no_grad():
            init_scores_t = self._safe_score(
                score_fn=score_fn,
                query_feat_1xc=query_feat_t,
                coords_raw_nx4=seeds_raw_t,
            )
        best_scores_np = init_scores_t.detach().cpu().numpy().reshape(-1)
        best_linear_np = means_linear_np.copy()
        no_improve_steps = np.zeros((n_start,), dtype=np.int32)

        active_start_indices = list(range(n_start))
        for _ in range(n_iterations):
            if len(active_start_indices) == 0:
                break

            # 1) Ask all starts.
            per_start_linear_np = []
            per_start_meta = []
            for start_idx in active_start_indices:
                opt = optimizers[start_idx]
                if enable_early_stop and self._optimizer_should_stop(opt):
                    continue
                candidates_i = np.stack([opt.ask() for _ in range(opt.population_size)], axis=0)
                per_start_linear_np.append(candidates_i)
                per_start_meta.append((start_idx, int(opt.population_size)))

            if len(per_start_linear_np) == 0:
                break

            # 2) Flatten and project bounds before evaluation.
            flat_linear_np = np.concatenate(per_start_linear_np, axis=0)
            flat_linear_t = self.project_linear_bounds(self._to_tensor(flat_linear_np))
            flat_linear_proj_np = flat_linear_t.detach().cpu().numpy()

            # 3) Convert to raw and evaluate in one big batch.
            flat_raw_t = self.processor.linear_to_raw(flat_linear_t)
            with torch.no_grad():
                flat_scores_t = self._safe_score(
                    score_fn=score_fn,
                    query_feat_1xc=query_feat_t,
                    coords_raw_nx4=flat_raw_t,
                )
            flat_scores_np = flat_scores_t.detach().cpu().numpy().reshape(-1)

            # 4) Tell each start and update local best.
            cursor = 0
            next_active_start_indices = []
            for start_idx, k in per_start_meta:
                opt = optimizers[start_idx]
                cand_linear_i = flat_linear_proj_np[cursor:cursor + k]
                scores_i = flat_scores_np[cursor:cursor + k]
                cursor += k

                # CMA minimizes objective.
                losses_i = -scores_i if maximize else scores_i
                losses_i = np.nan_to_num(losses_i, nan=1e9, posinf=1e9, neginf=-1e9)

                solutions = [(cand_linear_i[j], float(losses_i[j])) for j in range(k)]
                opt.tell(solutions)

                local_best_idx = int(np.argmax(scores_i)) if maximize else int(np.argmin(scores_i))
                local_best_score = scores_i[local_best_idx]
                if maximize:
                    is_better = local_best_score > best_scores_np[start_idx]
                else:
                    is_better = local_best_score < best_scores_np[start_idx]
                if is_better:
                    best_scores_np[start_idx] = local_best_score
                    best_linear_np[start_idx] = cand_linear_i[local_best_idx]

                if enable_early_stop:
                    if is_better:
                        no_improve_steps[start_idx] = 0
                    else:
                        no_improve_steps[start_idx] += 1

                if enable_early_stop and self._optimizer_should_stop(opt):
                    continue
                if enable_early_stop and no_improve_steps[start_idx] >= early_stop_patience:
                    continue
                next_active_start_indices.append(start_idx)

            active_start_indices = next_active_start_indices

        best_linear_t = self.project_linear_bounds(self._to_tensor(best_linear_np))
        best_raw_t = self.processor.linear_to_raw(best_linear_t)
        best_scores_t = self._to_tensor(best_scores_np).reshape(-1)
        return best_raw_t, best_scores_t

    def refine_batch_queries(
        self,
        query_feats,
        seed_coords_raw_batch,
        score_fn,
        score_pair_fn=None,
        sigma0=0.20,
        popsize=16,
        n_iterations=8,
        elite_ratio=None,
        maximize=True,
        cma_seed=None,
        cma_variant="CMA",
        enable_early_stop=False,
        early_stop_patience=3,
        return_diagnostics=False,
    ):
        """
        Batched multi-query wrapper around `refine_single_query`.

        Args:
            query_feats: [B, C] or [C]
            seed_coords_raw_batch: [B, K, 4] or [K, 4]
            score_fn: single-query score function kept for backward compatibility
            score_pair_fn: optional pairwise score function.
                Signature: (query_feats_mxc, coords_raw_mx4) -> [M]
                When provided, all active CMA populations across queries are
                evaluated in a single shared GPU batch.
        """
        seed_batch_t = self._to_tensor(seed_coords_raw_batch)
        q_feats_t = self._to_tensor(query_feats)

        if seed_batch_t.ndim == 2:
            return self.refine_single_query(
                query_feat=q_feats_t,
                seed_coords_raw=seed_batch_t,
                score_fn=score_fn,
                sigma0=sigma0,
                popsize=popsize,
                n_iterations=n_iterations,
                elite_ratio=elite_ratio,
                maximize=maximize,
                cma_seed=cma_seed,
                cma_variant=cma_variant,
                enable_early_stop=enable_early_stop,
                early_stop_patience=early_stop_patience,
            )

        if seed_batch_t.ndim != 3 or seed_batch_t.shape[-1] != 4:
            raise ValueError("seed_coords_raw_batch must be [B, K, 4] or [K, 4]")

        if q_feats_t.ndim == 1:
            q_feats_t = q_feats_t.unsqueeze(0)
        if q_feats_t.ndim != 2:
            raise ValueError("query_feats must be [B, C] or [C]")
        if q_feats_t.shape[0] != seed_batch_t.shape[0]:
            raise ValueError("query_feats batch size does not match seed_coords_raw_batch")

        self._ensure_cmaes_available()

        batch_size = int(seed_batch_t.shape[0])
        n_start = int(seed_batch_t.shape[1])
        if n_start == 0:
            empty_coords = torch.zeros((batch_size, 0, 4), device=self.device, dtype=torch.float32)
            empty_scores = torch.zeros((batch_size, 0), device=self.device, dtype=torch.float32)
            if return_diagnostics:
                return {
                    "best_coords": empty_coords,
                    "best_scores": empty_scores,
                    "final_population_coords": torch.zeros(
                        (batch_size, 0, 0, 4), device=self.device, dtype=torch.float32
                    ),
                    "final_population_scores": torch.zeros(
                        (batch_size, 0, 0), device=self.device, dtype=torch.float32
                    ),
                }
            return empty_coords, empty_scores

        popsize = max(4, int(popsize))
        n_iterations = max(1, int(n_iterations))
        sigma0 = max(1e-4, float(sigma0))
        cma_variant = self._normalize_cma_variant(cma_variant)
        use_lra = cma_variant == "LRA-CMA"
        enable_early_stop = bool(enable_early_stop)
        early_stop_patience = max(1, int(early_stop_patience))

        means_linear_t = self.processor.raw_to_linear(seed_batch_t.reshape(-1, 4)).reshape(batch_size, n_start, 4)
        means_linear_t = self.project_linear_bounds(means_linear_t)
        means_linear_np = means_linear_t.detach().cpu().numpy()

        bounds = np.array(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )

        optimizers = []
        for batch_idx in range(batch_size):
            opt_row = []
            for start_idx in range(n_start):
                seed_offset = batch_idx * 1000 + start_idx
                seed_i = None if cma_seed is None else int(cma_seed) + seed_offset
                cma_kwargs = dict(
                    mean=means_linear_np[batch_idx, start_idx].astype(np.float64),
                    sigma=sigma0,
                    bounds=bounds,
                    population_size=popsize,
                    seed=seed_i,
                )
                if use_lra:
                    cma_kwargs["lr_adapt"] = True
                try:
                    opt = CMA(**cma_kwargs)
                except TypeError as err:
                    if use_lra and "lr_adapt" in str(err):
                        raise RuntimeError(
                            "Current `cmaes` version does not support LRA-CMA (`lr_adapt`). "
                            "Please upgrade `cmaes` or switch cma_variant='CMA'."
                        ) from err
                    raise
                opt_row.append(opt)
            optimizers.append(opt_row)

        with torch.no_grad():
            init_scores_t = self._safe_score_pairs(
                score_pair_fn=score_pair_fn,
                score_fn=score_fn,
                query_feats_mxc=q_feats_t[:, None, :].expand(-1, n_start, -1).reshape(-1, q_feats_t.shape[-1]),
                coords_raw_mx4=seed_batch_t.reshape(-1, 4),
            )
        best_scores_np = init_scores_t.detach().cpu().numpy().reshape(batch_size, n_start)
        best_linear_np = means_linear_np.copy()
        no_improve_steps = np.zeros((batch_size, n_start), dtype=np.int32)
        active_pairs = [(batch_idx, start_idx) for batch_idx in range(batch_size) for start_idx in range(n_start)]

        final_population_linear_t = means_linear_t.unsqueeze(2).expand(-1, -1, popsize, -1).clone()
        final_population_scores_t = best_scores_np[:, :, None].repeat(popsize, axis=2)
        final_population_scores_t = torch.tensor(
            final_population_scores_t,
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(n_iterations):
            if len(active_pairs) == 0:
                break

            per_pair_linear_np = []
            per_pair_meta = []
            for batch_idx, start_idx in active_pairs:
                opt = optimizers[batch_idx][start_idx]
                if enable_early_stop and self._optimizer_should_stop(opt):
                    continue
                candidates_i = np.stack([opt.ask() for _ in range(opt.population_size)], axis=0)
                per_pair_linear_np.append(candidates_i)
                per_pair_meta.append((batch_idx, start_idx, int(opt.population_size)))

            if len(per_pair_linear_np) == 0:
                break

            flat_linear_np = np.concatenate(per_pair_linear_np, axis=0)
            flat_linear_t = self.project_linear_bounds(self._to_tensor(flat_linear_np))
            flat_linear_proj_np = flat_linear_t.detach().cpu().numpy()
            flat_raw_t = self.processor.linear_to_raw(flat_linear_t)

            query_index_list = []
            for batch_idx, _, k in per_pair_meta:
                query_index_list.extend([batch_idx] * k)
            query_index_t = torch.tensor(query_index_list, device=self.device, dtype=torch.long)
            flat_query_feats_t = q_feats_t.index_select(0, query_index_t)

            with torch.no_grad():
                flat_scores_t = self._safe_score_pairs(
                    score_pair_fn=score_pair_fn,
                    score_fn=score_fn,
                    query_feats_mxc=flat_query_feats_t,
                    coords_raw_mx4=flat_raw_t,
                )
            flat_scores_np = flat_scores_t.detach().cpu().numpy().reshape(-1)

            cursor = 0
            next_active_pairs = []
            for batch_idx, start_idx, k in per_pair_meta:
                opt = optimizers[batch_idx][start_idx]
                cand_linear_i = flat_linear_proj_np[cursor:cursor + k]
                scores_i_np = flat_scores_np[cursor:cursor + k]
                scores_i_t = flat_scores_t[cursor:cursor + k]
                cursor += k

                losses_i = -scores_i_np if maximize else scores_i_np
                losses_i = np.nan_to_num(losses_i, nan=1e9, posinf=1e9, neginf=-1e9)
                solutions = [(cand_linear_i[j], float(losses_i[j])) for j in range(k)]
                opt.tell(solutions)

                local_best_idx = int(np.argmax(scores_i_np)) if maximize else int(np.argmin(scores_i_np))
                local_best_score = scores_i_np[local_best_idx]
                if maximize:
                    is_better = local_best_score > best_scores_np[batch_idx, start_idx]
                else:
                    is_better = local_best_score < best_scores_np[batch_idx, start_idx]
                if is_better:
                    best_scores_np[batch_idx, start_idx] = local_best_score
                    best_linear_np[batch_idx, start_idx] = cand_linear_i[local_best_idx]

                final_population_linear_t[batch_idx, start_idx] = self._to_tensor(cand_linear_i)
                final_population_scores_t[batch_idx, start_idx] = scores_i_t

                if enable_early_stop:
                    if is_better:
                        no_improve_steps[batch_idx, start_idx] = 0
                    else:
                        no_improve_steps[batch_idx, start_idx] += 1

                if enable_early_stop and self._optimizer_should_stop(opt):
                    continue
                if enable_early_stop and no_improve_steps[batch_idx, start_idx] >= early_stop_patience:
                    continue
                next_active_pairs.append((batch_idx, start_idx))

            active_pairs = next_active_pairs

        best_linear_t = self.project_linear_bounds(self._to_tensor(best_linear_np.reshape(-1, 4))).reshape(
            batch_size, n_start, 4
        )
        best_raw_t = self.processor.linear_to_raw(best_linear_t.reshape(-1, 4)).reshape(batch_size, n_start, 4)
        best_scores_t = self._to_tensor(best_scores_np).reshape(batch_size, n_start)

        if not return_diagnostics:
            return best_raw_t, best_scores_t

        final_population_linear_t = self.project_linear_bounds(final_population_linear_t.reshape(-1, 4)).reshape(
            batch_size, n_start, popsize, 4
        )
        final_population_raw_t = self.processor.linear_to_raw(
            final_population_linear_t.reshape(-1, 4)
        ).reshape(batch_size, n_start, popsize, 4)
        return {
            "best_coords": best_raw_t,
            "best_scores": best_scores_t,
            "final_population_coords": final_population_raw_t,
            "final_population_scores": final_population_scores_t,
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _safe_score(self, score_fn, query_feat_1xc, coords_raw_nx4):
        """
        Normalize score_fn output to a finite torch vector [N].
        """
        scores = score_fn(query_feat_1xc, coords_raw_nx4)
        scores_t = self._to_tensor(scores).reshape(-1)
        scores_t = torch.nan_to_num(scores_t, nan=-1e9, posinf=1e9, neginf=-1e9)
        return scores_t

    def _safe_score_pairs(self, score_pair_fn, score_fn, query_feats_mxc, coords_raw_mx4):
        """
        Normalize pairwise score output to a finite torch vector [M].
        """
        if score_pair_fn is not None:
            scores = score_pair_fn(query_feats_mxc, coords_raw_mx4)
            scores_t = self._to_tensor(scores).reshape(-1)
            return torch.nan_to_num(scores_t, nan=-1e9, posinf=1e9, neginf=-1e9)

        score_chunks = []
        for idx in range(coords_raw_mx4.shape[0]):
            score_i = self._safe_score(
                score_fn=score_fn,
                query_feat_1xc=query_feats_mxc[idx:idx + 1],
                coords_raw_nx4=coords_raw_mx4[idx:idx + 1],
            )
            score_chunks.append(score_i.reshape(-1))
        if len(score_chunks) == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.float32)
        return torch.cat(score_chunks, dim=0)

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=torch.float32)
        return torch.tensor(x, device=self.device, dtype=torch.float32)

    @staticmethod
    def _ensure_cmaes_available():
        if _CMAES_AVAILABLE:
            return
        raise ImportError(
            "Package `cmaes` is not installed. Install it with: pip install cmaes"
        )
