import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union


class ProbabilisticProjector(nn.Module):
    """
    概率型流形投影器 (Probabilistic Manifold Projector)
    不仅将特征投影到低维流形，还同时预测该特征的不确定度 (Aleatoric Uncertainty)。

    Outputs:
        - embedding: 投影后的特征向量 (归一化到单位球) -> 用于计算余弦/欧氏距离
        - log_scale: 特征的不确定度 (Log Variance) -> 用于加权 Loss 或 拒绝预测
    """

    def __init__(
            self,
            input_dim: int = 1024,
            hidden_dims: List[int] = [512, 256],
            output_dim: int = 128,
            dropout: float = 0.0,
            predict_uncertainty: bool = True
    ):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty

        # === 1. Shared Backbone ===
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # === 2. Head A: Feature Projection ===
        self.feature_head = nn.Linear(curr_dim, output_dim)

        # === 3. Head B: Uncertainty Estimation ===
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(curr_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        # === 4. 执行初始化 (关键步骤) ===
        # 第一步：先应用通用的 Xavier/Kaiming 初始化到所有层
        self.apply(self._init_weights)

        # 第二步：覆盖 Uncertainty Head 的初始化
        # 我们希望初始状态下 log_scale 很小 (比如 -3.0, 对应 sigma ≈ 0.05)
        # 这样初始阶段网络会比较"自信"，主要依靠特征距离进行学习，避免 loss 被大的 sigma 摊平
        if self.predict_uncertainty:
            # 最后一个 Linear 层的 weight 设为 0，bias 设为负数
            nn.init.constant_(self.uncertainty_head[-1].weight, 0.0)
            nn.init.constant_(self.uncertainty_head[-1].bias, -3.0)

    def _init_weights(self, m):
        """
        通用初始化逻辑
        """
        if isinstance(m, nn.Linear):
            # Xavier Uniform 对于 Tanh/Sigmoid/Linear 较好
            # 对于 ReLU/GELU，Kaiming (He) Initialization 其实更好，
            # 但在 Metric Learning 中 Xavier 也很常用，保持和你之前一致即可。
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, D)
        Returns:
            embedding: (B, output_dim), L2 Normalized
            log_scale: (B, 1) or None
        """
        # 1. 提取共享特征
        feat = self.backbone(x)

        # 2. 计算特征向量 (均值)
        embedding = self.feature_head(feat)
        embedding = F.normalize(embedding, p=2, dim=-1)  # 强制投影到单位球

        # 3. 计算不确定度 (方差)
        log_scale = None
        if self.predict_uncertainty:
            log_scale = self.uncertainty_head(feat)  # (B, 1)
            # 限制范围防止数值不稳定 (可选)
            # log_scale = torch.clamp(log_scale, min=-10.0, max=5.0)

        return embedding, log_scale

    # def compute_energy_with_uncertainty(
    #         self,
    #         feat_q: torch.Tensor,
    #         feat_ref: torch.Tensor,
    #         metric: str = 'euclidean'
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    #     """
    #     计算带不确定度的能量（距离）。
    #
    #     Logic:
    #         Energy = Distance / Variance + Regularization
    #
    #     Args:
    #         feat_q: Query 特征 (B, D_in)
    #         feat_ref: Reference 特征 (B, K, D_in) 或 (B, D_in)
    #
    #     Returns:
    #         raw_energy: 原始的几何距离 (Euclidean/Cosine)
    #         uncertainty_term: 综合的不确定度 (log_scale_q + log_scale_ref)
    #     """
    #     # 1. Forward Pass
    #     emb_q, log_scale_q = self.forward(feat_q)  # (B, D_out), (B, 1)
    #     emb_ref, log_scale_ref = self.forward(feat_ref)  # (B, K, D_out), (B, K, 1)
    #
    #     # 2. 计算原始几何距离 (Raw Energy)
    #     # 这里复用之前的逻辑，支持 Batch Aligned 模式
    #     if metric == 'euclidean':
    #         if emb_ref.dim() == 3:
    #             # (B, 1, D) - (B, K, D)
    #             dist = torch.norm(emb_q.unsqueeze(1) - emb_ref, p=2, dim=-1)  # (B, K)
    #         else:
    #             dist = torch.norm(emb_q - emb_ref, p=2, dim=-1)  # (B,)
    #     elif metric == 'cosine':
    #         # Cosine Distance = 1 - Similarity
    #         if emb_ref.dim() == 3:
    #             sim = torch.einsum('bd,bkd->bk', emb_q, emb_ref)
    #             dist = 1.0 - sim
    #         else:
    #             sim = (emb_q * emb_ref).sum(dim=-1)
    #             dist = 1.0 - sim
    #     else:
    #         raise ValueError(f"Unknown metric: {metric}")
    #
    #     # 3. 处理不确定度
    #     if self.predict_uncertainty:
    #         # 综合不确定度: sigma_total^2 = sigma_q^2 + sigma_ref^2
    #         # 但为了数值稳定和简化计算，通常直接相加 Log Scale: log(s_total) ≈ log_s_q + log_s_ref
    #         # 广播机制: (B, 1) + (B, K, 1) -> (B, K, 1)
    #
    #         if emb_ref.dim() == 3:
    #             # log_scale_q: (B, 1) -> (B, 1, 1)
    #             total_log_scale = log_scale_q.unsqueeze(1) + log_scale_ref  # (B, K, 1)
    #             total_log_scale = total_log_scale.squeeze(-1)  # (B, K)
    #         else:
    #             total_log_scale = log_scale_q + log_scale_ref
    #             total_log_scale = total_log_scale.squeeze(-1)  # (B,)
    #
    #         return dist, total_log_scale
    #
    #     else:
    #         return dist, None

    def compute_energy_with_uncertainty(
            self,
            feat_q: torch.Tensor,
            feat_ref: torch.Tensor,
            metric: str = 'euclidean',
            pairwise: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算带不确定度的能量（距离）。

        Logic:
            Energy = Distance / Variance + Regularization

        Args:
            feat_q: Query 特征 (B, D_in) 或 (N, D_in)
            feat_ref: Reference 特征 (B, K, D_in), (B, D_in) 或 (M, D_in)
            metric: 'euclidean' 或 'cosine'
            pairwise: 是否强制开启 N x M 两两比对模式

        Returns:
            dist: 原始的几何距离 (Euclidean/Cosine)
                - Batch模式: (B,) 或 (B, K)
                - Pairwise模式: (N, M)
            log_scale: 综合的不确定度 (log_scale_q + log_scale_ref)
                - 形状与 dist 相同
        """
        # 1. Forward Pass
        emb_q, log_scale_q = self.forward(feat_q)  # (B, D_out), (B, 1) 或 (N, D_out), (N, 1)
        emb_ref, log_scale_ref = self.forward(feat_ref)  # (B, K, D_out), (B, K, 1) 或 (M, D_out), (M, 1)

        # 2. 判断是否需要 Pairwise 模式 (N x M 全量比对)
        is_cross_match = pairwise or (emb_q.shape[0] != emb_ref.shape[0] and emb_ref.dim() == 2)

        # 3. 计算原始几何距离 (Raw Energy)
        if is_cross_match:
            # Pairwise 模式: (N, D) vs (M, D) -> (N, M)
            if metric == 'euclidean':
                # (N, 1, D) - (1, M, D) -> (N, M, D) -> (N, M)
                dist = torch.norm(emb_q.unsqueeze(1) - emb_ref.unsqueeze(0), p=2, dim=-1)  # (N, M)
            elif metric == 'cosine':
                # (N, D) @ (D, M) -> (N, M)
                sim = torch.matmul(emb_q, emb_ref.t())
                dist = 1.0 - sim
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # 处理不确定度: (N, 1) + (1, M) -> (N, M)
            if self.predict_uncertainty:
                total_log_scale = log_scale_q + log_scale_ref.t()  # (N, 1) + (1, M) -> (N, M)
            else:
                total_log_scale = None

        else:
            # Batch Aligned 模式
            if metric == 'euclidean':
                if emb_ref.dim() == 3:
                    # (B, 1, D) - (B, K, D)
                    dist = torch.norm(emb_q.unsqueeze(1) - emb_ref, p=2, dim=-1)  # (B, K)
                else:
                    dist = torch.norm(emb_q - emb_ref, p=2, dim=-1)  # (B,)
            elif metric == 'cosine':
                # Cosine Distance = 1 - Similarity
                if emb_ref.dim() == 3:
                    sim = torch.einsum('bd,bkd->bk', emb_q, emb_ref)
                    dist = 1.0 - sim
                else:
                    sim = (emb_q * emb_ref).sum(dim=-1)
                    dist = 1.0 - sim
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # 处理不确定度
            if self.predict_uncertainty:
                # 综合不确定度: sigma_total^2 = sigma_q^2 + sigma_ref^2
                # 但为了数值稳定和简化计算，通常直接相加 Log Scale: log(s_total) ≈ log_s_q + log_s_ref
                # 广播机制: (B, 1) + (B, K, 1) -> (B, K, 1)

                if emb_ref.dim() == 3:
                    # log_scale_q: (B, 1) -> (B, 1, 1)
                    total_log_scale = log_scale_q.unsqueeze(1) + log_scale_ref  # (B, K, 1)
                    total_log_scale = total_log_scale.squeeze(-1)  # (B, K)
                else:
                    total_log_scale = log_scale_q + log_scale_ref
                    total_log_scale = total_log_scale.squeeze(-1)  # (B,)
            else:
                total_log_scale = None

        return dist, total_log_scale


# ============ 测试逻辑 ============
if __name__ == "__main__":
    # 初始化模型
    model = ProbabilisticProjector(
        input_dim=1024,
        hidden_dims=[512, 256],
        output_dim=128,
        predict_uncertainty=True
    )

    # 模拟数据
    B, K, D = 4, 100, 1024
    q = torch.randn(B, D)  # Query (Drone View)
    ref = torch.randn(B, K, D)  # Reference (Satellite Patches)

    # 计算
    raw_dist, log_scale = model.compute_energy_with_uncertainty(q, ref, metric='euclidean')

    print(f"Raw Distance Shape: {raw_dist.shape}")  # Should be (B, K)
    print(f"Log Scale Shape:    {log_scale.shape}")  # Should be (B, K)

    # 模拟 Loss 计算 (Kendall's Loss)
    # Loss = exp(-log_scale) * dist + log_scale
    # 注意: dist 这里应该是 loss 本身 (比如 L2 Loss 或 CrossEntropy 的 logits)
    # 在你的场景下，这会变成 Weighted Softmax

    temperature = torch.exp(log_scale)
    print(f"Predicted Temperature (Mean): {temperature.mean().item():.4f}")