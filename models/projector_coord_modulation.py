import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    稳定版 FiLM
    特性：
    1. 限制 Gamma 范围，防止流形剧烈撕裂
    2. 显式处理 Gamma 的中心偏移 (1 + ...)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, raw_gamma, raw_beta):
        # x: [B, C]
        # raw_gamma, raw_beta: [B, C]

        # 1. 限制 Gamma 范围: [1-alpha, 1+alpha]
        # 这样保证了乘数因子永远是正数（当 alpha <= 1），且不会过大
        gamma = 1.0 + self.alpha * torch.tanh(raw_gamma)

        # 2. Beta 可以不限制，或者也限制，通常不限制 Beta 问题不大
        beta = raw_beta

        # 广播机制处理维度
        if x.dim() == 3 and gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return x * gamma + beta


class ConditionalResBlock(nn.Module):
    """
    【增强版】几何调制残差块
    改进点：
    1. Pre-Norm: 在 FiLM 之前标准化特征
    2. Constrained FiLM: 使用稳定版 FiLM
    3. Zero-Init Last Linear: 最后一层初始化为0，确保初始状态为纯 Identity
    """

    def __init__(self, dim, dropout=0.1, film_alpha=1.0):
        super().__init__()

        # Pre-Norm 层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 全连接层
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # 稳定版 FiLM
        self.film = FiLM(alpha=film_alpha)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        # FC1: Kaiming
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        # FC2: Zero Init!
        # 这是一个非常强的 Trick。让残差分支初始输出为0。
        # 这样整个 Block 初始就是一个 Identity Mapping (x = x + 0)。
        # 配合 Gamma=1, Beta=0，训练极其稳定。
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, gammas, betas):
        """
        Args:
            x: [B, dim]
            gammas: [raw_gamma1, raw_gamma2]
            betas: [raw_beta1, raw_beta2]
        """
        identity = x

        # Block 1
        # Pre-Norm -> FiLM -> Act -> Dropout -> Linear
        out = self.norm1(x)
        out = self.film(out, gammas[0], betas[0])
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc1(out)

        # Block 2
        out = self.norm2(out)
        out = self.film(out, gammas[1], betas[1])
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out + identity


class CoordModulator(nn.Module):
    """
    配合稳定版 FiLM 的参数生成器
    """

    def __init__(self, coord_dim, feat_dim, num_res_blocks):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.feat_dim = feat_dim

        # ... (中间层同前) ...
        self.net = nn.Sequential(
            nn.Linear(coord_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_res_blocks * 2 * 2 * feat_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # 这里改了！
        # 因为我们在 FiLM 里写了 gamma = 1 + tanh(raw)，
        # 所以这里的 raw_gamma 应该初始化为 0，而不是 1。
        # tanh(0) = 0 -> gamma = 1。
        # beta 初始化为 0。
        # 所以最后一层全部初始化为 0 即可。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coord_emb):
        B = coord_emb.shape[0]
        # [B, total_params]
        params = self.net(coord_emb)
        # Reshape: [B, Blocks, 2(Layers), 2(Type), Dim]
        params = params.view(B, self.num_res_blocks, 2, 2, self.feat_dim)
        return params


class GeoModulatedProjector(nn.Module):
    """
    【最终简化版】几何调制投影器
    输入: 视觉特征 (Query), 坐标编码 (Condition)
    输出: 匹配能量分数 (Energy Score)

    原理:
    网络作为一个函数 E(I, x)，通过 x 生成的参数去调制 I 的前向传播。
    如果 I 和 x 匹配，网络输出低能量（或高分）；如果不匹配，输出高能量。
    """

    def __init__(self,
                 feat_dim=1024,
                 coord_dim=128,
                 hidden_dim=512,
                 num_res_blocks=3,
                 dropout=0.1):
        super().__init__()

        # 1. 视觉特征降维/预处理
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 坐标调制器 (生成所有层的 FiLM 参数)
        self.modulator = CoordModulator(coord_dim, hidden_dim, num_res_blocks)

        # 3. 条件残差主体
        self.blocks = nn.ModuleList([
            ConditionalResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])

        # 4. 能量读出头 (Readout Head)
        # 输出 1 维标量。
        # 如果是 Energy 模型，越小越好；如果是 Logits，越大越好。
        # 这里建议输出 Logits (Similarity)，后续可以用 BCEWithLogitsLoss
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # 归一化有助于稳定能量场
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_query, coord_ref_encoded):
        """
        feat_query: [B, Feat_Dim] (无人机图像特征)
        coord_ref_encoded: [B, Coord_Dim] (候选坐标)
        """
        # 1. 准备视觉特征
        x = self.feat_proj(feat_query)  # [B, Hidden]

        # 2. 准备调制参数
        # params: [B, Blocks, 2, 2, Hidden]
        params = self.modulator(coord_ref_encoded)

        # 3. 逐层调制穿行
        for i, block in enumerate(self.blocks):
            # 取出当前 Block 所需的参数
            # gammas: [gamma_layer1, gamma_layer2]
            # betas: [beta_layer1, beta_layer2]
            block_params = params[:, i]
            gammas = [block_params[:, 0, 0], block_params[:, 1, 0]]
            betas = [block_params[:, 0, 1], block_params[:, 1, 1]]

            x = block(x, gammas, betas)

        # 4. 读出分数
        score = self.head(x)  # [B, 1]

        return score.squeeze(-1)



# ==========================================
# 2. Main 函数 (结构分析与测试)
# ==========================================

def count_parameters(model, name="Model"):
    """辅助函数：统计并打印参数量"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 [{name}] Total Parameters: {total_params / 1e6:.2f} M (百万)")
    return total_params


def main():
    print("=" * 50)
    print("🚀 Initializing GeoModulatedProjector...")
    print("=" * 50)

    # 1. 配置参数
    config = {
        'feat_dim': 1024,  # 无人机视觉特征维度 (Backbone输出)
        'coord_dim': 128,  # 坐标位置编码维度 (x,y,s,d -> PE)
        'hidden_dim': 512,  # 内部流形维度
        'num_res_blocks': 3,  # 深度
        'dropout': 0.1
    }

    # 2. 实例化模型
    model = GeoModulatedProjector(**config)

    # 3. 打印网络结构
    print(model)
    print("-" * 50)

    # 4. 统计参数量
    total = count_parameters(model, "Full Projector")

    # 细分统计 (看看参数主要分布在哪里)
    print("\n🧐 Breakdown:")
    count_parameters(model.feat_proj, "Visual Projection")
    count_parameters(model.modulator, "Coord Modulator")
    count_parameters(model.blocks, "ResBlocks Body")

    print("-" * 50)

    # 5. 模拟一次前向传播 (Sanity Check)
    batch_size = 4

    # 模拟输入数据
    dummy_feat = torch.randn(batch_size, config['feat_dim'])  # [4, 1024]
    dummy_coord = torch.randn(batch_size, config['coord_dim'])  # [4, 128]

    print(f"📥 Input Shapes:")
    print(f"   - Query Feature: {dummy_feat.shape}")
    print(f"   - Coord Encoded: {dummy_coord.shape}")

    # 推理
    output_score = model(dummy_feat, dummy_coord)

    print(f"\n📤 Output Shape: {output_score.shape}")
    print(f"   - Values (Logits): {output_score.detach().numpy()}")

    # 6. 验证 Zero-Init 效果
    # 理论上，由于最后一层全零初始化，初始输出应该在 LayerNorm 和 Bias 的作用下接近 0 或某个常数
    print(f"\n✅ Initial Stability Check:")
    print(f"   Mean Score: {output_score.mean().item():.4f}")
    print(f"   Std  Score: {output_score.std().item():.4f} (Should be small)")


if __name__ == "__main__":
    main()