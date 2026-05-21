import logging

import torch


def _silence_evotorch_logs():
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    logging.getLogger("evotorch.core").setLevel(logging.WARNING)


try:
    from evotorch import Problem
    from evotorch.algorithms import CMAES

    _EVOTORCH_AVAILABLE = True
    _silence_evotorch_logs()
except Exception:
    Problem = None
    CMAES = None
    _EVOTORCH_AVAILABLE = False


class MultiStartCMAESEvoTorchRefiner:
    """
    Multi-start CMA-ES refiner backed by EvoTorch.

    Coordinate convention:
        raw   : [nr, nc, theta_rad, scale]
        linear: [nr_n, nc_n, theta_lin, scale_n], each nominally in [-1, 1]
    """

    def __init__(self, coords_processor, device=None):
        self.processor = coords_processor
        self.device = device if device is not None else coords_processor.device

    @staticmethod
    def _wrap_theta_linear(theta_lin):
        return torch.remainder(theta_lin + 1.0, 2.0) - 1.0

    def project_linear_bounds(self, coords_linear):
        coords = coords_linear.clone()
        coords[..., 0:2] = coords[..., 0:2].clamp(-1.0, 1.0)
        coords[..., 2:3] = self._wrap_theta_linear(coords[..., 2:3])
        coords[..., 3:4] = coords[..., 3:4].clamp(-1.0, 1.0)
        return coords

    @staticmethod
    def cyclic_rot_diff_deg(theta_a, theta_b):
        diff = torch.abs(theta_a - theta_b)
        diff = torch.minimum(diff, 2 * torch.pi - diff)
        return diff * 180.0 / torch.pi

    def select_diverse_seeds(
        self,
        coords_sorted,
        scores_sorted,
        target_k,
        enable_seed_dedup=True,
        dedup_radius_nrc=0.08,
        dedup_radius_deg=10.0,
        dedup_scale_rel=0.11,
    ):
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

    @staticmethod
    def _normalize_cma_variant(cma_variant):
        variant = str(cma_variant).strip().lower()
        if variant in ("cma", "standard", "base"):
            return "CMA"
        if variant in ("sep-cma", "separable", "sep", "diag"):
            return "Sep-CMA"
        raise ValueError(f"Unsupported cma_variant: {cma_variant}. Use 'CMA' or 'Sep-CMA'.")

    @staticmethod
    def _compute_h_sig(searcher):
        d = searcher.p_sigma.shape[-1]
        squared_sum = torch.norm(searcher.p_sigma).pow(2.0) / (
            1 - (1 - searcher.c_sigma) ** (2 * searcher.step_count + 1)
        )
        stall = (squared_sum / d) - 1 < 1 + 4.0 / (d + 1)
        return stall.any().to(searcher.p_sigma.dtype)

    def _project_partial_linear(self, xs, fixed_linear, optimize_dims):
        full = fixed_linear.reshape(1, 4).expand(xs.shape[0], 4).clone()
        full[:, optimize_dims] = xs
        full = self.project_linear_bounds(full)
        xs_proj = full[:, optimize_dims]
        return xs_proj, full

    def _project_search_samples(self, searcher, xs):
        if hasattr(searcher, "optimize_dims"):
            xs_proj, full_proj = self._project_partial_linear(
                xs,
                fixed_linear=searcher.fixed_linear_full,
                optimize_dims=searcher.optimize_dims,
            )
        else:
            xs_proj = self.project_linear_bounds(xs)
            full_proj = xs_proj
        ys_proj = (xs_proj - searcher.m.unsqueeze(0)) / searcher.sigma
        if searcher.separable:
            zs_proj = ys_proj / searcher.A.unsqueeze(0).clamp(min=1e-12)
        else:
            zs_proj = torch.linalg.solve(searcher.A, ys_proj.T).T
        return zs_proj, ys_proj, xs_proj, full_proj

    def _make_problem(self, solution_length=4, seed=None):
        return Problem(
            "min",
            solution_length=int(solution_length),
            initial_bounds=(-1.0, 1.0),
            dtype=torch.float32,
            device=self.device,
            seed=seed,
        )

    def _make_searcher(self, center_init, sigma0, popsize, cma_variant, seed=None):
        problem = self._make_problem(solution_length=int(center_init.numel()), seed=seed)
        return CMAES(
            problem,
            stdev_init=sigma0,
            popsize=popsize,
            center_init=center_init,
            separable=(cma_variant == "Sep-CMA"),
        )

    @staticmethod
    def _regularize_covariance_matrix(cov: torch.Tensor, eps: float) -> torch.Tensor:
        cov_reg = 0.5 * (cov + cov.transpose(-1, -2))
        cov_reg = torch.nan_to_num(cov_reg, nan=0.0, posinf=0.0, neginf=0.0)
        dim = cov_reg.shape[-1]
        eye = torch.eye(dim, device=cov_reg.device, dtype=cov_reg.dtype)

        if not torch.isfinite(cov_reg).all():
            return eye * eps

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov_reg)
            eigvals = torch.nan_to_num(eigvals, nan=eps, posinf=eps, neginf=eps).clamp(min=eps)
            cov_reg = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.transpose(-1, -2)
            cov_reg = 0.5 * (cov_reg + cov_reg.transpose(-1, -2))
            return cov_reg + eps * eye
        except RuntimeError:
            diag = torch.diagonal(cov_reg, dim1=-2, dim2=-1)
            min_diag = float(diag.min().item()) if diag.numel() > 0 else 0.0
            if min_diag < eps:
                cov_reg = cov_reg + (eps - min_diag) * eye
            return cov_reg + eps * eye

    def _ensure_searcher_factorization(self, searcher) -> None:
        if getattr(searcher, "separable", False):
            searcher.A = searcher.A.clamp(min=1e-8)
            return

        last_err = None
        for eps in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            try:
                searcher.C = self._regularize_covariance_matrix(searcher.C, eps=eps)
                searcher.decompose_C()
                return
            except RuntimeError as err:
                last_err = err
        dim = int(searcher.C.shape[-1])
        eye = torch.eye(dim, device=searcher.C.device, dtype=searcher.C.dtype)
        try:
            searcher.C = eye.clone()
            searcher.decompose_C()
            return
        except RuntimeError as err:
            last_err = err
        if last_err is not None:
            raise last_err

    def refine_single_query(
        self,
        query_feat,
        seed_coords_raw,
        score_pair_fn,
        sigma0=0.20,
        popsize=16,
        n_iterations=8,
        maximize=True,
        cma_seed=None,
        cma_variant="CMA",
        enable_early_stop=False,
        early_stop_patience=3,
        return_diagnostics=False,
        optimize_dims=None,
    ):
        result = self.refine_batch_queries(
            query_feats=self._to_tensor(query_feat).reshape(1, -1),
            seed_coords_raw_batch=self._to_tensor(seed_coords_raw).reshape(1, -1, 4),
            score_pair_fn=score_pair_fn,
            sigma0=sigma0,
            popsize=popsize,
            n_iterations=n_iterations,
            maximize=maximize,
            cma_seed=cma_seed,
            cma_variant=cma_variant,
            enable_early_stop=enable_early_stop,
            early_stop_patience=early_stop_patience,
            return_diagnostics=return_diagnostics,
            optimize_dims=optimize_dims,
        )
        if return_diagnostics:
            return {
                "best_coords": result["best_coords"].squeeze(0),
                "best_scores": result["best_scores"].squeeze(0),
                "final_population_coords": result["final_population_coords"].squeeze(0),
                "final_population_scores": result["final_population_scores"].squeeze(0),
            }
        coords, scores = result
        return coords.squeeze(0), scores.squeeze(0)

    def refine_batch_queries(
        self,
        query_feats,
        seed_coords_raw_batch,
        score_pair_fn,
        sigma0=0.20,
        popsize=16,
        n_iterations=8,
        maximize=True,
        cma_seed=None,
        cma_variant="CMA",
        enable_early_stop=False,
        early_stop_patience=3,
        return_diagnostics=False,
        optimize_dims=None,
    ):
        self._ensure_evotorch_available()

        seed_batch_t = self._to_tensor(seed_coords_raw_batch)
        q_feats_t = self._to_tensor(query_feats)
        if seed_batch_t.ndim == 2:
            seed_batch_t = seed_batch_t.unsqueeze(0)
        if q_feats_t.ndim == 1:
            q_feats_t = q_feats_t.unsqueeze(0)

        if seed_batch_t.ndim != 3 or seed_batch_t.shape[-1] != 4:
            raise ValueError("seed_coords_raw_batch must be [B, K, 4] or [K, 4]")
        if q_feats_t.ndim != 2:
            raise ValueError("query_feats must be [B, C] or [C]")
        if q_feats_t.shape[0] != seed_batch_t.shape[0]:
            raise ValueError("query_feats batch size does not match seed_coords_raw_batch")

        batch_size = int(seed_batch_t.shape[0])
        n_start = int(seed_batch_t.shape[1])
        if n_start == 0:
            empty_coords = torch.zeros((batch_size, 0, 4), device=self.device, dtype=torch.float32)
            empty_scores = torch.zeros((batch_size, 0), device=self.device, dtype=torch.float32)
            if return_diagnostics:
                return {
                    "best_coords": empty_coords,
                    "best_scores": empty_scores,
                    "final_population_coords": torch.zeros((batch_size, 0, 0, 4), device=self.device),
                    "final_population_scores": torch.zeros((batch_size, 0, 0), device=self.device),
                }
            return empty_coords, empty_scores

        popsize = max(4, int(popsize))
        n_iterations = max(1, int(n_iterations))
        cma_variant = self._normalize_cma_variant(cma_variant)
        enable_early_stop = bool(enable_early_stop)
        early_stop_patience = max(1, int(early_stop_patience))
        sigma0_t = self._resolve_sigma0_batch(
            sigma0=sigma0,
            batch_size=batch_size,
            n_start=n_start,
        )
        optimize_dims = tuple(int(d) for d in ((0, 1, 2, 3) if optimize_dims is None else optimize_dims))
        if len(optimize_dims) <= 0 or any(d < 0 or d > 3 for d in optimize_dims):
            raise ValueError(f"optimize_dims must be a non-empty subset of [0,1,2,3], got {optimize_dims}")

        means_linear_full_t = self.processor.raw_to_linear(seed_batch_t.reshape(-1, 4)).reshape(batch_size, n_start, 4)
        means_linear_full_t = self.project_linear_bounds(means_linear_full_t)
        means_linear_t = means_linear_full_t[..., optimize_dims]

        searchers = []
        for batch_idx in range(batch_size):
            searcher_row = []
            for start_idx in range(n_start):
                seed_offset = batch_idx * 1000 + start_idx
                seed_i = None if cma_seed is None else int(cma_seed) + seed_offset
                searcher = self._make_searcher(
                    center_init=means_linear_t[batch_idx, start_idx],
                    sigma0=float(sigma0_t[batch_idx, start_idx].item()),
                    popsize=popsize,
                    cma_variant=cma_variant,
                    seed=seed_i,
                )
                searcher.optimize_dims = optimize_dims
                searcher.fixed_linear_full = means_linear_full_t[batch_idx, start_idx].clone()
                self._ensure_searcher_factorization(searcher)
                searcher_row.append(searcher)
            searchers.append(searcher_row)

        with torch.no_grad():
            init_scores_t = self._safe_score_pairs(
                score_pair_fn=score_pair_fn,
                query_feats_mxc=q_feats_t[:, None, :].expand(-1, n_start, -1).reshape(-1, q_feats_t.shape[-1]),
                coords_raw_mx4=seed_batch_t.reshape(-1, 4),
            )
        best_scores_t = init_scores_t.reshape(batch_size, n_start)
        best_linear_t = means_linear_t.clone()
        no_improve_steps = torch.zeros((batch_size, n_start), dtype=torch.long, device=self.device)
        active_pairs = [(batch_idx, start_idx) for batch_idx in range(batch_size) for start_idx in range(n_start)]

        final_population_linear_t = means_linear_full_t.unsqueeze(2).expand(-1, -1, popsize, -1).clone()
        final_population_scores_t = best_scores_t.unsqueeze(-1).expand(-1, -1, popsize).clone()

        for _ in range(n_iterations):
            if len(active_pairs) == 0:
                break

            sampled_linear_parts = []
            sampled_full_linear_parts = []
            sampled_meta = []
            sampled_internal = []
            for batch_idx, start_idx in active_pairs:
                searcher = searchers[batch_idx][start_idx]
                zs_i, ys_i, xs_i = searcher.sample_distribution()
                zs_proj_i, ys_proj_i, xs_proj_i, xs_full_proj_i = self._project_search_samples(searcher, xs_i)
                sampled_internal.append((zs_proj_i, ys_proj_i))
                sampled_linear_parts.append(xs_proj_i)
                sampled_full_linear_parts.append(xs_full_proj_i)
                sampled_meta.append((batch_idx, start_idx, searcher.popsize))

            if len(sampled_linear_parts) == 0:
                break

            flat_linear_t = torch.cat(sampled_linear_parts, dim=0)
            flat_full_linear_t = torch.cat(sampled_full_linear_parts, dim=0)
            flat_raw_t = self.processor.linear_to_raw(flat_full_linear_t)
            query_index_t = torch.tensor(
                [batch_idx for batch_idx, _, k in sampled_meta for _ in range(k)],
                device=self.device,
                dtype=torch.long,
            )
            flat_query_feats_t = q_feats_t.index_select(0, query_index_t)

            with torch.no_grad():
                flat_scores_t = self._safe_score_pairs(
                    score_pair_fn=score_pair_fn,
                    query_feats_mxc=flat_query_feats_t,
                    coords_raw_mx4=flat_raw_t,
                )

            cursor = 0
            next_active_pairs = []
            for meta_idx, (batch_idx, start_idx, k) in enumerate(sampled_meta):
                searcher = searchers[batch_idx][start_idx]
                zs_i, ys_i = sampled_internal[meta_idx]
                xs_i = flat_linear_t[cursor:cursor + k]
                scores_i = flat_scores_t[cursor:cursor + k]
                cursor += k

                losses_i = -scores_i if maximize else scores_i
                searcher.population.set_values(xs_i)
                searcher.population.set_evals(losses_i)
                indices = searcher.population.argsort(obj_index=searcher.obj_index)
                ranks = torch.zeros_like(indices)
                ranks[indices] = torch.arange(searcher.popsize, dtype=indices.dtype, device=indices.device)
                assigned_weights = searcher.weights[ranks]

                local_m_displacement, shaped_m_displacement = searcher.update_m(zs_i, ys_i, assigned_weights)
                searcher.update_p_sigma(local_m_displacement)
                searcher.update_sigma()
                h_sig = self._compute_h_sig(searcher)
                searcher.update_p_c(shaped_m_displacement, h_sig)
                searcher.update_C(zs_i, ys_i, assigned_weights, h_sig)
                if (searcher.step_count + 1) % searcher.decompose_C_freq == 0:
                    self._ensure_searcher_factorization(searcher)
                searcher._steps_count += 1

                if maximize:
                    local_best_idx = torch.argmax(scores_i)
                    is_better = scores_i[local_best_idx] > best_scores_t[batch_idx, start_idx]
                else:
                    local_best_idx = torch.argmin(scores_i)
                    is_better = scores_i[local_best_idx] < best_scores_t[batch_idx, start_idx]
                if bool(is_better):
                    best_scores_t[batch_idx, start_idx] = scores_i[local_best_idx]
                    best_linear_t[batch_idx, start_idx] = xs_i[local_best_idx]
                    no_improve_steps[batch_idx, start_idx] = 0
                elif enable_early_stop:
                    no_improve_steps[batch_idx, start_idx] += 1

                final_population_linear_t[batch_idx, start_idx] = flat_full_linear_t[cursor - k:cursor]
                final_population_scores_t[batch_idx, start_idx] = scores_i

                if enable_early_stop and no_improve_steps[batch_idx, start_idx] >= early_stop_patience:
                    continue
                next_active_pairs.append((batch_idx, start_idx))

            active_pairs = next_active_pairs

        best_linear_full_t = means_linear_full_t.clone()
        best_linear_full_t[..., optimize_dims] = best_linear_t
        best_linear_full_t = self.project_linear_bounds(best_linear_full_t.reshape(-1, 4)).reshape(batch_size, n_start, 4)
        best_raw_t = self.processor.linear_to_raw(best_linear_full_t.reshape(-1, 4)).reshape(batch_size, n_start, 4)
        best_scores_t = torch.nan_to_num(best_scores_t, nan=-1e9, posinf=1e9, neginf=-1e9)

        if not return_diagnostics:
            return best_raw_t, best_scores_t

        final_population_linear_t = self.project_linear_bounds(final_population_linear_t.reshape(-1, 4)).reshape(
            batch_size, n_start, popsize, 4
        )
        final_population_raw_t = self.processor.linear_to_raw(
            final_population_linear_t.reshape(-1, 4)
        ).reshape(batch_size, n_start, popsize, 4)
        final_population_scores_t = torch.nan_to_num(
            final_population_scores_t,
            nan=-1e9,
            posinf=1e9,
            neginf=-1e9,
        )
        return {
            "best_coords": best_raw_t,
            "best_scores": best_scores_t,
            "final_population_coords": final_population_raw_t,
            "final_population_scores": final_population_scores_t,
        }

    def _safe_score_pairs(self, score_pair_fn, query_feats_mxc, coords_raw_mx4):
        scores = score_pair_fn(query_feats_mxc, coords_raw_mx4)
        scores_t = self._to_tensor(scores).reshape(-1)
        return torch.nan_to_num(scores_t, nan=-1e9, posinf=1e9, neginf=-1e9)

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.detach().clone().to(device=self.device, dtype=torch.float32)
        return torch.tensor(x, device=self.device, dtype=torch.float32)

    def _resolve_sigma0_batch(self, sigma0, batch_size: int, n_start: int) -> torch.Tensor:
        if torch.is_tensor(sigma0):
            sigma0_t = sigma0.detach().clone().to(device=self.device, dtype=torch.float32)
        else:
            sigma0_t = torch.tensor(sigma0, device=self.device, dtype=torch.float32)

        if sigma0_t.ndim == 0:
            sigma0_t = sigma0_t.reshape(1, 1).expand(batch_size, n_start).clone()
        elif sigma0_t.ndim == 1:
            if int(sigma0_t.numel()) == n_start:
                sigma0_t = sigma0_t.reshape(1, n_start).expand(batch_size, n_start).clone()
            elif int(sigma0_t.numel()) == batch_size:
                sigma0_t = sigma0_t.reshape(batch_size, 1).expand(batch_size, n_start).clone()
            else:
                raise ValueError(
                    f"sigma0 with shape {tuple(sigma0_t.shape)} is incompatible with batch_size={batch_size}, n_start={n_start}."
                )
        elif sigma0_t.ndim == 2:
            if tuple(sigma0_t.shape) != (batch_size, n_start):
                raise ValueError(
                    f"sigma0 with shape {tuple(sigma0_t.shape)} must match (batch_size, n_start)=({batch_size}, {n_start})."
                )
        else:
            raise ValueError("sigma0 must be a scalar, [K], [B], or [B, K].")

        sigma0_t = torch.nan_to_num(sigma0_t, nan=1e-4, posinf=1.0, neginf=1e-4)
        return sigma0_t.clamp(min=1e-4)

    @staticmethod
    def _ensure_evotorch_available():
        if _EVOTORCH_AVAILABLE:
            return
        raise ImportError(
            "Package `evotorch` is not installed in the current python environment."
        )
