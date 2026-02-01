import numpy as np
import torch

from .coord_distance import SceneNegMasker, neg_mask_visloc


class ANNIndex:
    # 功能: 构建/查询近邻索引，提供 faiss 与暴力检索两种后端。
    def __init__(self, dim, backend="faiss", use_gpu=False, metric="l2"):
        self.dim = int(dim)
        self.backend = backend
        self.use_gpu = use_gpu
        self.metric = metric
        self.index = None
        self._feats = None

    def build(self, feats):
        feats_np = np.ascontiguousarray(feats, dtype=np.float32)
        if feats_np.ndim != 2 or feats_np.shape[1] != self.dim:
            raise ValueError(f"feats must be [N,{self.dim}] float32")

        if self.backend == "faiss":
            try:
                import faiss
            except Exception:
                self.backend = "bruteforce"
                self._feats = feats_np
                self.index = None
                return

            if self.metric == "l2":
                index = faiss.IndexFlatL2(self.dim)
            elif self.metric in ("ip", "cosine"):
                index = faiss.IndexFlatIP(self.dim)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            index.add(feats_np)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            self.index = index
            self._feats = None
        elif self.backend == "bruteforce":
            self._feats = feats_np
            self.index = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def search(self, feats_q, k):
        feats_q = np.ascontiguousarray(feats_q, dtype=np.float32)
        if feats_q.ndim != 2 or feats_q.shape[1] != self.dim:
            raise ValueError(f"feats_q must be [B,{self.dim}] float32")

        if k <= 0:
            raise ValueError("k must be > 0")

        if self.backend == "faiss":
            if self.index is None:
                raise RuntimeError("Index not built")
            dists, indices = self.index.search(feats_q, k)
            return dists, indices

        if self._feats is None:
            raise RuntimeError("Index not built")

        g = self._feats
        k = min(k, g.shape[0])
        if self.metric == "l2":
            q_norm = (feats_q ** 2).sum(axis=1, keepdims=True)
            g_norm = (g ** 2).sum(axis=1)
            dist = q_norm + g_norm[None, :] - 2.0 * feats_q @ g.T
            idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = np.arange(dist.shape[0])[:, None]
            dist_top = dist[row, idx]
            order = np.argsort(dist_top, axis=1)
            idx = idx[row, order]
            dist_top = dist_top[row, order]
            return dist_top, idx
        if self.metric in ("ip", "cosine"):
            sim = feats_q @ g.T
            idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row = np.arange(sim.shape[0])[:, None]
            sim_top = sim[row, idx]
            order = np.argsort(-sim_top, axis=1)
            idx = idx[row, order]
            sim_top = sim_top[row, order]
            return sim_top, idx
        raise ValueError(f"Unknown metric: {self.metric}")


class ANCENegativeMiner:
    # 功能: 基于 ANN 结果过滤正样本半径内候选，输出 hard negatives 坐标。
    def __init__(self, index: ANNIndex, neg_masker: SceneNegMasker | None = None):
        self.index = index
        self.coords_gallery = None
        self.neg_masker = neg_masker or SceneNegMasker()

    @property
    def gallery_size(self):
        return 0 if self.coords_gallery is None else int(self.coords_gallery.shape[0])

    def update_gallery(self, feats_gallery, coords_gallery):
        coords = torch.as_tensor(coords_gallery, dtype=torch.float32, device="cpu")
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("coords_gallery must be [N,4] or [N,>=2]")
        self.coords_gallery = coords
        if feats_gallery is not None:
            self.index.build(feats_gallery)

    def _sample_random_negatives(
        self,
        q_coord,
        n_needed,
        radius_nrc,
        radius_rot_rad=None,
        max_tries=5,
    ):
        if n_needed <= 0:
            return torch.empty(0, dtype=torch.long)
        n_total = self.coords_gallery.shape[0]
        picked = []
        tries = 0
        while len(picked) == 0 or (torch.cat(picked).numel() < n_needed and tries < max_tries):
            tries += 1
            n_sample = max(n_needed * 4, 32)
            idx = torch.randint(0, n_total, (n_sample,), dtype=torch.long)
            mask = self.neg_masker.neg_mask(
                self.coords_gallery[idx],
                q_coord,
                radius_nrc=radius_nrc,
                radius_rot_rad=radius_rot_rad,
            )
            idx = idx[mask]
            if idx.numel() > 0:
                picked.append(idx)
        if not picked:
            return torch.empty(0, dtype=torch.long)
        idx_all = torch.cat(picked)[:n_needed]
        return idx_all

    def mine(
        self,
        query_feats,
        query_coords,
        top_k=100,
        n_neg=10,
        radius_nrc=0.0,
        radius_rot_rad=None,
        random_negatives=True,
        return_indices=False,
    ):
        if self.coords_gallery is None:
            raise RuntimeError("coords_gallery not set, call update_gallery first")

        n_total = self.coords_gallery.shape[0]
        top_k = min(int(top_k), n_total)
        if top_k <= 0:
            raise ValueError("top_k must be > 0 and <= gallery size")

        q_coords = torch.as_tensor(query_coords, dtype=torch.float32, device="cpu")
        if q_coords.ndim != 2 or q_coords.shape[1] < 2:
            raise ValueError("query_coords must be [B,4] or [B,>=2]")

        dists, indices = self.index.search(query_feats, top_k)
        idx_t = torch.from_numpy(indices).to(dtype=torch.long)

        coords_topk = self.coords_gallery[idx_t]  # [B,K,4]
        mask = self.neg_masker.neg_mask(
            coords_topk,
            q_coords,
            radius_nrc=radius_nrc,
            radius_rot_rad=radius_rot_rad,
        )

        neg_indices = []
        for i in range(q_coords.shape[0]):
            cand = idx_t[i][mask[i]]
            if cand.numel() < n_neg:
                need = n_neg - cand.numel()
                if random_negatives:
                    extra = self._sample_random_negatives(
                        q_coords[i],
                        need,
                        radius_nrc,
                        radius_rot_rad,
                    )
                    cand = torch.cat([cand, extra], dim=0)
            if cand.numel() < n_neg:
                extra = torch.randint(0, n_total, (n_neg - cand.numel(),), dtype=torch.long)
                cand = torch.cat([cand, extra], dim=0)
            neg_indices.append(cand[:n_neg])

        neg_indices = torch.stack(neg_indices, dim=0)  # [B, n_neg]
        neg_coords = self.coords_gallery[neg_indices]  # [B, n_neg, 4]

        if return_indices:
            return neg_coords, neg_indices
        return neg_coords


class MultiSceneANCEMiner:
    # 功能: 管理多场景的 ANCE 矿工实例，各场景独立索引与挖掘。
    def __init__(
        self,
        backend="faiss",
        use_gpu=False,
        metric="l2",
        maskers_by_scene=None,
        default_masker: SceneNegMasker | None = None,
    ):
        self.backend = backend
        self.use_gpu = use_gpu
        self.metric = metric
        self.maskers_by_scene = maskers_by_scene or {}
        self.default_masker = default_masker
        self.miners = {}

    def update_scene(self, scene_name, feats_gallery, coords_gallery):
        if scene_name not in self.miners:
            dim = feats_gallery.shape[1]
            index = ANNIndex(dim, backend=self.backend, use_gpu=self.use_gpu, metric=self.metric)
            masker = self.maskers_by_scene.get(scene_name, None)
            if masker is None:
                masker = self.default_masker or SceneNegMasker()
            self.miners[scene_name] = ANCENegativeMiner(index, neg_masker=masker)
        self.miners[scene_name].update_gallery(feats_gallery, coords_gallery)

    def mine(self, scene_name, *args, **kwargs):
        if scene_name not in self.miners:
            raise KeyError(f"Scene '{scene_name}' not found in miners")
        return self.miners[scene_name].mine(*args, **kwargs)
