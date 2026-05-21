import torch
import torch.nn.functional as TF
import torch.nn.functional as F

def _compute_metric_from_query_and_points(
        self,
        query_feats,
        ref_points,
        temperature=10.0,
        metric='possibility',
        coord_space='raw',
        chunk_size=2048,
        feat_type='projector',
    ):
        """
        计算 query_feats 与 ref_points 的距离或概率。

        Args:
            query_feats: [B, C]
            ref_points: [N, 4] or [B, N, 4]
            temperature: softmax 温度（仅在 metric='possibility' 时使用）
            metric: 'dist' 返回距离；'possibility' 返回 softmax 概率。
                    兼容旧写法: 'euclidean'/'l2' 等同于 'possibility'
            coord_space: 'raw' (nr,nc,rot,scale) 或 'linear'
            chunk_size: 候选分块大小
            feat_type: 'ingp' 使用 INGP 特征距离；'projector' 使用 Projector 输出特征距离
        Returns:
            dist 或 prob: [B, N]
        """
        metric = metric.lower()
        if metric in ('euclidean', 'l2'):
            metric = 'possibility'
        if metric not in ('dist', 'possibility'):
            raise ValueError(f"metric must be 'dist' or 'possibility', got {metric}")

        feat_type = feat_type.lower()
        if feat_type not in ('ingp', 'projector'):
            raise ValueError(f"feat_type must be 'ingp' or 'projector', got {feat_type}")

        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # 如果是形如 [N, 1, 4] 的输入，视为单batch展开
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])
        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points shape not supported after reshape: {ref_points.shape}")

        # 归一化坐标
        if coord_space == 'linear':
            ref_points_raw = self.coord_normer.linear_to_raw(ref_points.reshape(-1, 4))
        else:
            ref_points_raw = ref_points.reshape(-1, 4)

        # chunk_size 控制候选维度分块，避免 grid_mlp 激活过大
        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points_raw.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        # 预处理 query 特征
        if feat_type == 'ingp':
            query_emb = TF.normalize(query_feats, dim=-1)
        else:
            with torch.no_grad():
                query_emb = self.projector(query_feats)

        dist_chunks = []
        if ref_points.dim() == 2:
            for start in range(0, ref_points_raw.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points_raw.shape[0])
                coords_chunk = ref_points_raw[start:end]
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1)
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)
        else:
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coords_chunk_flat = coords_chunk.reshape(-1, 4)
                if feat_type == 'ingp':
                    feats_ref = self._get_feats_fm_INGP(coords_chunk_flat, coord_mode='raw')
                    feats_ref = TF.normalize(feats_ref, dim=-1).view(B, -1, feats_ref.shape[-1])
                else:
                    coords_6d = self.coord_normer.raw_to_net(coords_chunk_flat, append_linear_rot=True)
                    grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                    feats_grid_raw = self._get_feats_fm_grid(grid_input)
                    coords_encoded_stage2 = self.pos_encoder_grid(coords_6d[:, :5])
                    feats_ingp = self.grid_mlp(feats_grid_raw, coords_encoded_stage2)
                    feats_ref = self.projector(TF.normalize(feats_ingp, dim=-1))
                    feats_ref = feats_ref.view(B, -1, feats_ref.shape[-1])
                if feat_type == 'ingp':
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                else:
                    dist_chunk = torch.norm(query_emb.unsqueeze(1) - feats_ref, p=2, dim=-1)
                dist_chunks.append(dist_chunk)

        dist = torch.cat(dist_chunks, dim=1)

        if metric == 'dist':
            return dist

        logit = -temperature * dist
        prob = F.softmax(logit, dim=-1)
        return prob

def _compute_metric_from_ingp(
        self,
        query_feats,
        ref_points,
        coord_space='raw',
        chunk_size=4096,
        metric='sim',
    ):
        """
        使用 INGP 特征与视觉特征计算相似度/距离，支持分块避免 OOM。

        Returns:
            metric_out: [B, N] 相似度或距离
        """
        if coord_space not in ('raw', 'linear'):
            raise ValueError(f"coord_space must be 'raw' or 'linear', got {coord_space}")

        if ref_points.dim() not in (2, 3):
            raise ValueError(f"ref_points must be 2D or 3D, got {ref_points.dim()}D")

        # 展平不匹配 batch 的 [N,1,4] 等情况
        if ref_points.dim() == 3 and ref_points.shape[0] != query_feats.shape[0]:
            ref_points = ref_points.reshape(-1, ref_points.shape[-1])

        q_norm = TF.normalize(query_feats, dim=-1)
        metric = metric.lower()
        if metric not in ('sim', 'dist'):
            raise ValueError(f"metric must be 'sim' or 'dist', got {metric}")

        if chunk_size is None or chunk_size <= 0:
            chunk_size = ref_points.shape[0] if ref_points.dim() == 2 else ref_points.shape[1]

        metric_chunks = []
        if ref_points.dim() == 2:
            # 公用候选 [N, 4]
            for start in range(0, ref_points.shape[0], chunk_size):
                end = min(start + chunk_size, ref_points.shape[0])
                coords_chunk = ref_points[start:end]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk, coord_mode=coord_mode)
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    # [B, C] x [C, chunk] -> [B, chunk]
                    metric_chunk = torch.matmul(q_norm, feats_chunk.t())
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        elif ref_points.dim() == 3:
            # 每个样本各自的候选 [B, N, 4]
            B = ref_points.shape[0]
            for start in range(0, ref_points.shape[1], chunk_size):
                end = min(start + chunk_size, ref_points.shape[1])
                coords_chunk = ref_points[:, start:end, :]  # [B, chunk, 4]
                coord_mode = 'raw' if coord_space == 'raw' else 'linear'
                feats_chunk = self._get_feats_fm_INGP(coords_chunk.reshape(-1, 4), coord_mode=coord_mode)
                feats_chunk = feats_chunk.view(B, -1, feats_chunk.shape[-1])
                feats_chunk = TF.normalize(feats_chunk, dim=-1)
                if metric == 'sim':
                    metric_chunk = torch.sum(q_norm.unsqueeze(1) * feats_chunk, dim=-1)  # [B, chunk]
                else:
                    metric_chunk = torch.norm(q_norm.unsqueeze(1) - feats_chunk, p=2, dim=-1)
                metric_chunks.append(metric_chunk)
        else:
            raise ValueError(f"ref_points shape not supported: {ref_points.shape}")

        metric_out = torch.cat(metric_chunks, dim=1)
        return metric_out

def _get_feats_fm_INGP(self, coords, coord_mode='raw'):
        """
        [通用原子操作] 输入任意格式坐标，输出归一化特征

        修改为与 stage3_project_integrateRot_classify.py 一致的实现方式

        Args:
            coords: 坐标张量，形状可以是 [N, 4] 或 [N, 5] 或 [N, 6]
            coord_mode: 输入坐标的类型
                - 'raw':     [N, 4] 物理坐标 [r, c, theta, s] (Dataset输出)
                - 'linear':  [N, 4] 线性坐标 [r_n, c_n, t_lin, s_n] (Sampler输出)
                - 'net_5d':  [N, 5] 网络坐标 [r_n, c_n, cos, sin, s_n] (直接透传)
                - 'net_6d':  [N, 6] 网络坐标+线性角度 (Processor输出的中间态)

        Returns:
            feat_norm: [N, C] L2归一化后的特征
        """

        # =========================================================
        # 1. 统一转换层：转换为 6D 格式 (与 stage3_project_integrateRot_classify 一致)
        # =========================================================
        if coord_mode == 'raw':
            # 直接使用 raw_to_net 并追加 linear_rot
            coords_6d = self.coord_normer.raw_to_net(coords, append_linear_rot=True)
        elif coord_mode == 'linear':
            coords_net = self.coord_normer.linear_to_net(coords)
            theta_lin = coords[..., 2:3]  # linear 空间的 theta 已经是归一化的
            coords_6d = torch.cat([coords_net, theta_lin], dim=-1)
        elif coord_mode == 'net_5d':
            theta_lin = torch.atan2(coords[..., 3:4], coords[..., 2:3]) / torch.pi
            coords_6d = torch.cat([coords, theta_lin], dim=-1)
        elif coord_mode == 'net_6d':
            coords_6d = coords
        else:
            raise ValueError(f"Unknown coord_mode: {coord_mode}")

        # =========================================================
        # 2. 构造子模块输入 (与 stage3_project_integrateRot_classify 一致)
        # =========================================================
        grid_input = torch.cat([coords_6d[..., :2], coords_6d[..., -1:]], dim=-1)  # [nr, nc, theta_lin]
        mlp_input = coords_6d[..., :5]  # [nr, nc, cos, sin, log_s]

        # =========================================================
        # 3. 前向传播
        # =========================================================

        # (A) Query HashGrid (Backbone)
        feat_raw = self._get_feats_fm_grid(grid_input)

        # (B) Positional Encoding (用于 MLP 条件)
        pos_enc = self.pos_encoder_grid(mlp_input)

        # (C) Tiny MLP (Decoder)
        feat_out = self.grid_mlp(feat_raw, pos_enc)

        # 4. L2 Normalize (Metric Learning 标准操作)
        return torch.nn.functional.normalize(feat_out, dim=-1, p=2)
