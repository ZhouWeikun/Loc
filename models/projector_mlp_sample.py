import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class Projector(nn.Module):
    """
    灵活的高维流形投影器 (Flexible Manifold Projector)
    支持自定义深度的 MLP 结构，将视觉/神经场特征投影到低维单位球面上。

    Architecture:
        Input -> [Linear->LayerNorm->GELU->Dropout] x N -> Linear -> L2Norm

    Args:
        input_dim (int): 输入特征维度
        hidden_dims (List[int]): 隐藏层维度列表。
                                 例如 [512, 256] 表示两层隐藏层。
                                 如果为空 []，则变为单层线性映射。
        output_dim (int): 输出特征维度
        dropout (float): 隐藏层的 Dropout 概率，默认 0.0
    """

    def __init__(
            self,
            input_dim: int = 1024,
            hidden_dims: List[int] = [512, 256],
            output_dim: int = 128,
            dropout: float = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # === 构建隐藏层块 ===
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            # LayerNorm 对 Transformer/NeRF 类特征通常比 BatchNorm 更稳健
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            curr_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)

        # === 输出层 ===
        # 注意：输出层前通常不再加 Activation，直接映射到流形空间
        self.output_layer = nn.Linear(curr_dim, output_dim)

        # === 初始化 ===
        self._init_weights()

    def _init_weights(self):
        """权重初始化：Xavier Uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, input_dim) 或 (B, N, input_dim)
            normalize: 是否输出单位向量 (project onto unit sphere)
        """
        # 1. 通过所有隐藏层
        x = self.hidden_layers(x)

        # 2. 最终线性投影
        x = self.output_layer(x)

        # 3. L2 归一化 (投影到超球面上)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def project_pair(
            self,
            feat_q: torch.Tensor,
            feat_ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """辅助函数：同时处理一对特征"""
        return self.forward(feat_q), self.forward(feat_ref)

    def compute_energy(
            self,
            feat_q: torch.Tensor,
            feat_ref: torch.Tensor,
            metric: str = 'cosine',
            pairwise: bool = False
    ) -> torch.Tensor:
        """
        计算能量/距离，自动适配 Batch 模式或 Pairwise 模式。

        Args:
            feat_q: shape (B, D) 或 (N, D)
            feat_ref: shape (B, D), (B, K, D) 或 (M, D)
            metric: 'cosine' 或 'euclidean'
            pairwise: 是否强制开启 N x M 两两比对模式。
                      如果输入 shape[0] 不一致，会自动开启此模式。

        Returns:
            Energy tensor.
            - Batch模式: (B,) 或 (B, K)
            - Pairwise模式: (N, M)
        """
        # 1. 投影到流形空间 (L2 Normalized)
        q_proj = self.forward(feat_q, normalize=True)
        ref_proj = self.forward(feat_ref, normalize=True)

        # 2. 判断是否需要全量两两比对 (N vs M)
        # 条件：显式指定 pairwise=True，或者 Batch 维度不匹配 (意味着不能一一对应)
        is_cross_match = pairwise or (q_proj.shape[0] != ref_proj.shape[0])

        if metric == 'cosine':
            if is_cross_match:
                # === Case A: N x M Cross Matching ===
                # q: (N, D), ref: (M, D) -> output: (N, M)
                # Cosine 相似度就是矩阵乘法 (因为已经归一化了)
                # 确保 ref 是 2D，如果是 (1, M, D) 这种 3D 需要 squeeze
                if ref_proj.dim() == 3:
                    ref_proj = ref_proj.reshape(-1, ref_proj.shape[-1])

                return torch.mm(q_proj, ref_proj.t())

            else:
                # === Case B: Batch Aligned ===
                # q: (B, D), ref: (B, K, D) -> output: (B, K)
                if ref_proj.dim() == 3:
                    return torch.einsum('bd,bnd->bn', q_proj, ref_proj)
                # q: (B, D), ref: (B, D) -> output: (B,)
                else:
                    return (q_proj * ref_proj).sum(dim=-1)

        elif metric == 'euclidean':
            if is_cross_match:
                # === Case A: N x M Cross Matching ===
                # 使用 torch.cdist 高效计算两组向量集合的距离矩阵
                if ref_proj.dim() == 3:
                    ref_proj = ref_proj.reshape(-1, ref_proj.shape[-1])

                # p=2 代表欧氏距离
                return torch.cdist(q_proj, ref_proj, p=2)

            else:
                # === Case B: Batch Aligned ===
                if ref_proj.dim() == 3:
                    # (B, 1, D) - (B, K, D) -> (B, K, D) -> norm
                    return torch.norm(q_proj.unsqueeze(1) - ref_proj, p=2, dim=-1)
                else:
                    return torch.norm(q_proj - ref_proj, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============ 测试用例 ============

if __name__ == "__main__":
    # 配置 A：复刻你之前的 "瓶颈" 结构
    # Input(1024) -> 64 -> 128
    model_bottleneck = Projector(
        input_dim=1024,
        hidden_dims=[64],  # 只有一个隐藏层
        output_dim=128
    )
    print(f"Model A (Bottleneck) Params: {model_bottleneck.num_parameters}")

    # 配置 B：渐进式压缩 (类似 ResNet/ViT head)
    # Input(1024) -> 512 -> 256 -> 128
    model_deep = Projector(
        input_dim=1024,
        hidden_dims=[512, 256],
        output_dim=128,
        dropout=0.1
    )
    print(f"Model B (Deep) Params: {model_deep.num_parameters}")

    # 打印网络结构看一下
    print("\n--- Deep Model Structure ---")
    print(model_deep)

    # 运行一次数据流测试
    B, N, D = 2, 100, 1024
    q = torch.randn(B, D)
    ref = torch.randn(B, N, D)

    energy = model_deep.compute_energy(q, ref, metric='cosine')
    print(f"\nForward Pass Check: Energy shape {energy.shape} (Expected: {B}, {N})")