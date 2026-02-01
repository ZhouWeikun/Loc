import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ResBlock(nn.Module):
    """
    带谱归一化的残差块 (Spectral Normalized Residual Block)
    结构: Input -> [Linear -> LN -> GELU -> Dropout] -> Linear -> Add -> Output
    """

    def __init__(self, dim, expansion_factor=2, dropout=0.0, use_spectral_norm=True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        # 定义线性层构建函数，可选谱归一化
        def make_linear(in_d, out_d):
            layer = nn.Linear(in_d, out_d)
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        self.net = nn.Sequential(
            make_linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            make_linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # 如果使用了谱归一化，通常不需要LayerNorm在最后，但加一个LayerNorm有助于训练稳定
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.final_ln(x + self.net(x))


class AdvancedProjector(nn.Module):
    """
    改进版流形投影器
    特性：
    1. 支持 Residual Connection (深层网络更易训练)
    2. 支持 Spectral Normalization (强制流形平滑，利于梯度下降)
    3. 灵活的 Expansion (反向瓶颈)
    """

    def __init__(
            self,
            input_dim: int = 1024,
            output_dim: int = 128,
            num_res_blocks: int = 2,  # 残差块数量
            hidden_dim: int = 1024,  # 残差块内部维度 (建议 >= input_dim)
            expansion_factor: int = 2,  # 残差块内的升维倍数
            dropout: float = 0.1,
            use_spectral_norm: bool = False  # 开启谱归一化以获得平滑流形
    ):
        super().__init__()

        # 1. 维度对齐层 (如果 hidden_dim != input_dim)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # 2. 残差主体 (ResMLP)
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResBlock(
                    dim=hidden_dim,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                    use_spectral_norm=use_spectral_norm
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # 3. 输出投影层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 如果开启谱归一化，对输出层也应用
        if use_spectral_norm:
            self.output_layer = nn.utils.spectral_norm(self.output_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Kaiming 初始化通常比 Xavier 更适合深层网络
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # Input Projection
        x = self.input_proj(x)

        # Residual Blocks
        x = self.blocks(x)

        # Output Projection
        x = self.output_layer(x)

        # L2 Normalization (Project to Sphere)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    # 保留原本好用的 compute_energy 接口
    def compute_energy(self, feat_q, feat_ref, metric='cosine', pairwise=False):
        # ... (完全复用你之前的代码逻辑) ...
        # 这里为了节省篇幅略去，逻辑与你原版一致
        q_proj = self.forward(feat_q, normalize=True)
        ref_proj = self.forward(feat_ref, normalize=True)
        is_cross_match = pairwise or (q_proj.shape[0] != ref_proj.shape[0])

        if metric == 'cosine':
            if is_cross_match:
                if ref_proj.dim() == 3: ref_proj = ref_proj.reshape(-1, ref_proj.shape[-1])
                return torch.mm(q_proj,
                                ref_proj.t())  # Cosine distance is 1 - similarity, but usually for logits we use similarity directly
            else:
                if ref_proj.dim() == 3:
                    return torch.einsum('bd,bnd->bn', q_proj, ref_proj)
                else:
                    return (q_proj * ref_proj).sum(dim=-1)
        elif metric == 'euclidean':
            # ... 同你之前的逻辑 ...
            if is_cross_match:
                if ref_proj.dim() == 3: ref_proj = ref_proj.reshape(-1, ref_proj.shape[-1])
                return torch.cdist(q_proj, ref_proj, p=2)
            else:
                if ref_proj.dim() == 3:
                    return torch.norm(q_proj.unsqueeze(1) - ref_proj, p=2, dim=-1)
                else:
                    return torch.norm(q_proj - ref_proj, p=2, dim=-1)


# 使用示例
if __name__ == "__main__":
    # 初始化一个更强的 Projector
    # 假设输入 1024，我们保持 1024 维度做 2 层 ResBlock 处理，内部升维到 2048，最后降维到 128
    projector = AdvancedProjector(
        input_dim=1024,
        hidden_dim=1024,  # 保持宽通道
        expansion_factor=2,  # 内部升维到 2048
        num_res_blocks=2,  # 深度
        output_dim=128,
        use_spectral_norm=True  # 关键：开启谱归一化
    )

    print(f"Params: {sum(p.numel() for p in projector.parameters())}")

    x = torch.randn(4, 1024)
    out = projector(x)
    print("Output shape:", out.shape)
    print("Norm check:", torch.norm(out, dim=-1))