import torch
import torch.nn as nn
from models.res_mlp import ResnetBlockFC


class BranchMLP(nn.Module):
    """特征交互分支的降维MLP: 1024→512（中间激活）"""

    def __init__(self, input_dim: int = 1024,
                 hidden_dim: int = 768,
                 output_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 1024]
        Returns:
            [B, N, 512]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CoordToFiLM(nn.Module):
    """坐标编码转FiLM调制参数"""

    def __init__(self, coord_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 num_branches: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.output_dim = output_dim

        # MLP: coord_dim → hidden → hidden → 3×2×512
        self.fc1 = nn.Linear(coord_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_branches * 2 * output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, coord_encoded: torch.Tensor) -> tuple:
        """
        Args:
            coord_encoded: [B, N, D_coord]
        Returns:
            gamma: [B, N, 3, 512]
            beta: [B, N, 3, 512]
        """
        B, N, _ = coord_encoded.shape

        # Forward through MLP
        x = self.relu(self.fc1(coord_encoded))
        x = self.relu(self.fc2(x))
        params = self.fc3(x)  # [B, N, 3×2×512]

        # Reshape to [B, N, 3, 2, 512]
        params = params.view(B, N, self.num_branches, 2, self.output_dim)

        # Split into gamma and beta
        gamma = params[:, :, :, 0, :]  # [B, N, 3, 512]
        beta = params[:, :, :, 1, :]  # [B, N, 3, 512]

        return gamma, beta


class FusionMLP(nn.Module):
    """最终融合MLP: 768→1（无激活输出）"""

    def __init__(self, input_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 768]
        Returns:
            [B, N, 1]
        """
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 无激活
        return x


class MetricNet(nn.Module):
    """
    可学习的距离度量网络，用于预测查询特征与参考特征之间的不匹配代价

    Args:
        feat_dim: 输入特征维度（默认1024）
        coord_dim: 坐标编码维度
        branch_hidden_dim: Branch MLP的隐藏层维度
        branch_output_dim: Branch MLP的输出维度（调制前）
        resblock_hidden_dim: 残差块的隐藏层维度
        resblock_output_dim: 残差块的输出维度（调制后）
        dropout: Dropout率
        init_weights: 是否使用自定义权重初始化
    """

    def __init__(self,
                 feat_dim: int = 1024,
                 coord_dim: int = 128,
                 branch_hidden_dim: int = 512,
                 branch_output_dim: int = 512,
                 resblock_hidden_dim: int = 384,
                 resblock_output_dim: int = 256,
                 dropout: float = 0.1,
                 init_weights: bool = True,
                 output_activation: nn.Module = None,
                 ):
        super().__init__()

        self.feat_dim = feat_dim
        self.coord_dim = coord_dim
        self.branch_output_dim = branch_output_dim
        self.resblock_output_dim = resblock_output_dim
        # 保存激活函数
        self.output_activation = output_activation  # <--- 2. 保存

        # ============ 阶段2: Branch MLPs (1024→512) ============
        self.branch_mul = BranchMLP(
            feat_dim, branch_hidden_dim, branch_output_dim, dropout
        )
        self.branch_sub = BranchMLP(
            feat_dim, branch_hidden_dim, branch_output_dim, dropout
        )
        self.branch_sq = BranchMLP(
            feat_dim, branch_hidden_dim, branch_output_dim, dropout
        )

        # ============ 阶段3: CoordToFiLM ============
        self.coord_to_film = CoordToFiLM(
            coord_dim=coord_dim,
            hidden_dim=256,
            output_dim=branch_output_dim,
            num_branches=3
        )

        # ============ 阶段4: 预激活残差块 (512→256) ============
        self.resblock_mul = ResnetBlockFC(
            input_dim=branch_output_dim,
            output_dim=resblock_output_dim,
            hidden_dim=resblock_hidden_dim,
            norm_type='none',
            activation_type='relu',
            init_type='resnet_zero_init'
        )
        self.resblock_sub = ResnetBlockFC(
            input_dim=branch_output_dim,
            output_dim=resblock_output_dim,
            hidden_dim=resblock_hidden_dim,
            norm_type='none',
            activation_type='relu',
            init_type='resnet_zero_init'
        )
        self.resblock_sq = ResnetBlockFC(
            input_dim=branch_output_dim,
            output_dim=resblock_output_dim,
            hidden_dim=resblock_hidden_dim,
            norm_type='none',
            activation_type='relu',
            init_type='resnet_zero_init'
        )

        # ============ 阶段5: 融合MLP (768→1) ============
        self.fusion_mlp = FusionMLP(
            input_dim=resblock_output_dim * 3,
            dropout=dropout
        )

        # ============ 权重初始化 ============
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        """
        自定义权重初始化策略

        策略：
        1. BranchMLP: Kaiming Normal (适配ReLU)
        2. CoordToFiLM:
           - 中间层: Kaiming Normal
           - 最后一层: 特殊初始化（gamma→1, beta→0）
        3. FusionMLP: Kaiming Normal
        4. ResnetBlockFC: 保持原有的resnet_zero_init（不覆盖）
        """

        # ============ 初始化 BranchMLP ============
        for branch in [self.branch_mul, self.branch_sub, self.branch_sq]:
            for name, module in branch.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # ============ 初始化 CoordToFiLM ============
        # 中间层: Kaiming初始化
        nn.init.kaiming_normal_(self.coord_to_film.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.coord_to_film.fc1.bias)
        nn.init.kaiming_normal_(self.coord_to_film.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.coord_to_film.fc2.bias)

        # 最后一层: 特殊初始化（FiLM语义）
        # 目标: gamma初始≈1, beta初始≈0
        # 输出维度: 3 branches × 2 (gamma,beta) × 512 dim = 3072
        num_branches = self.coord_to_film.num_branches
        output_dim = self.coord_to_film.output_dim

        # 权重: 小随机初始化
        nn.init.normal_(self.coord_to_film.fc3.weight, mean=0.0, std=0.01)

        # bias: 分段初始化
        with torch.no_grad():
            bias = self.coord_to_film.fc3.bias
            # Reshape to [3, 2, 512]
            bias_reshaped = bias.view(num_branches, 2, output_dim)

            # gamma部分 (index 0): 初始化为1
            bias_reshaped[:, 0, :] = 1.0

            # beta部分 (index 1): 初始化为0
            bias_reshaped[:, 1, :] = 0.0

            # 写回
            bias.copy_(bias_reshaped.view(-1))

        # ============ 初始化 FusionMLP ============
        for name, module in self.fusion_mlp.named_modules():
            if isinstance(module, nn.Linear):
                # 最后一层用更小的初始化（输出层）
                if name == 'fc4':
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # ============ ResnetBlockFC 保持原有初始化 ============
        # 不覆盖，因为ResnetBlockFC已经有resnet_zero_init
        pass


    def forward(self,
                feat_query: torch.Tensor,
                feat_ref: torch.Tensor,
                coord_ref_encoded: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 自适应维度版本

        Args:
            feat_query: 查询特征
                - [B, N, C] → 输出 [B, N]
                - [M, C] → 输出 [M]
            feat_ref: 参考特征
            coord_ref_encoded: 编码后的参考坐标

        Returns:
            distance: 预测的不匹配代价，维度与输入对应
        """
        # ============ 步骤1: 检测输入维度并保存信息 ============
        input_is_2d = (feat_ref.dim() == 2)

        if input_is_2d:
            # 输入是 [M, C]，转换为 [1, M, C]
            feat_query = feat_query.unsqueeze(0)  # [M, C] → [1, M, C]
            feat_ref = feat_ref.unsqueeze(0)  # [M, C] → [1, M, C]
            coord_ref_encoded = coord_ref_encoded.unsqueeze(0)  # [M, D] → [1, M, D]

        # ============ 步骤2: 现在输入统一是 [B, N, C] 格式 ============
        B, N, C = feat_ref.shape

        # 如果feat_query是[B, C]，扩展到[B, N, C]
        if feat_query.dim() == 2:
            feat_query = feat_query.unsqueeze(1).expand(B, N, C)

        # ============ 阶段1-5: 原有处理逻辑（完全不变）============
        # 阶段1: 特征交互 (1024维)
        h_mul = feat_query * feat_ref
        h_sub = feat_query - feat_ref
        h_sq = (feat_query - feat_ref) ** 2

        # 阶段2: Branch降维 (1024→512)
        h_mul_512 = self.branch_mul(h_mul)
        h_sub_512 = self.branch_sub(h_sub)
        h_sq_512 = self.branch_sq(h_sq)

        # 阶段3: FiLM调制
        gamma, beta = self.coord_to_film(coord_ref_encoded)
        # gamma, beta: [B, N, 3, 512]

        h_mul_mod = gamma[:, :, 0, :] * h_mul_512 + beta[:, :, 0, :]
        h_sub_mod = gamma[:, :, 1, :] * h_sub_512 + beta[:, :, 1, :]
        h_sq_mod = gamma[:, :, 2, :] * h_sq_512 + beta[:, :, 2, :]

        # 阶段4: 预激活残差降维 (512→256)
        h_mul_256 = self.resblock_mul(h_mul_mod)
        h_sub_256 = self.resblock_sub(h_sub_mod)
        h_sq_256 = self.resblock_sq(h_sq_mod)

        # 阶段5: 拼接与融合 (768→1)
        h_concat = torch.cat([h_mul_256, h_sub_256, h_sq_256], dim=-1)
        distance = self.fusion_mlp(h_concat)  # [B, N, 1]

        # ============ [核心修改] 在这里应用激活 ============
        distance = self.output_activation(distance) if self.output_activation is not None else distance

        # ============ 步骤3: 恢复到原始输入格式 ============
        distance = distance.squeeze(-1)  # [B, N, 1] → [B, N]

        if input_is_2d:
            # 如果原始输入是2D，squeeze掉batch维度
            distance = distance.squeeze(0)  # [1, M] → [M]

        return distance

    def predict_batch(self,
                      feat_query: torch.Tensor,
                      feat_refs: torch.Tensor,
                      coord_refs_encoded: torch.Tensor) -> torch.Tensor:
        """
        批量预测（便捷接口）

        Args:
            feat_query: 单个查询特征 [C] 或 [1, C]
            feat_refs: 多个参考特征 [N, C]
            coord_refs_encoded: 多个参考坐标编码 [N, D_coord]

        Returns:
            distances: [N, 1] 或 [N]
        """
        # 添加batch维度
        if feat_query.dim() == 1:
            feat_query = feat_query.unsqueeze(0)  # [1, C]
        if feat_refs.dim() == 2:
            feat_refs = feat_refs.unsqueeze(0)  # [1, N, C]
        if coord_refs_encoded.dim() == 2:
            coord_refs_encoded = coord_refs_encoded.unsqueeze(0)  # [1, N, D_coord]

        with torch.no_grad():
            distances = self.forward(feat_query, feat_refs, coord_refs_encoded)

        return distances.squeeze(0)  # [N, 1]

    def get_film_params_stats(self) -> dict:
        """
        获取FiLM参数的统计信息（用于监控训练）

        Returns:
            stats: 包含gamma和beta统计的字典
        """
        # 需要一个dummy输入来获取FiLM参数
        # 这个方法主要用于训练过程中的监控
        return {
            'coord_to_film_fc3_bias_mean': self.coord_to_film.fc3.bias.mean().item(),
            'coord_to_film_fc3_bias_std': self.coord_to_film.fc3.bias.std().item(),
            'coord_to_film_fc3_weight_mean': self.coord_to_film.fc3.weight.mean().item(),
            'coord_to_film_fc3_weight_std': self.coord_to_film.fc3.weight.std().item(),
        }


# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始化网络（启用自定义初始化）
    metric_net = MetricNet(
        feat_dim=1024,
        coord_dim=128,
        branch_hidden_dim=768,
        branch_output_dim=512,
        resblock_hidden_dim=384,
        resblock_output_dim=256,
        dropout=0.1,
        init_weights=True  # 启用自定义初始化
    )

    # 打印参数量
    total_params = sum(p.numel() for p in metric_net.parameters())
    trainable_params = sum(p.numel() for p in metric_net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 检查FiLM初始化
    print("\n=== FiLM Initialization Check ===")
    with torch.no_grad():
        bias = metric_net.coord_to_film.fc3.bias
        bias_reshaped = bias.view(3, 2, 512)

        gamma_bias = bias_reshaped[:, 0, :]  # [3, 512]
        beta_bias = bias_reshaped[:, 1, :]  # [3, 512]

        print(f"Gamma bias - mean: {gamma_bias.mean():.4f}, std: {gamma_bias.std():.4f}")
        print(f"Beta bias - mean: {beta_bias.mean():.4f}, std: {beta_bias.std():.4f}")
        print(f"Expected: gamma≈1.0, beta≈0.0")

    # 测试前向传播
    print("\n=== Forward Pass Test ===")
    B, N, C = 4, 32, 1024
    D_coord = 128

    feat_query = torch.randn(B, C)
    feat_ref = torch.randn(B, N, C)
    coord_encoded = torch.randn(B, N, D_coord)

    distances = metric_net(feat_query, feat_ref, coord_encoded)
    print(f"Input shapes:")
    print(f"  feat_query: {feat_query.shape}")
    print(f"  feat_ref: {feat_ref.shape}")
    print(f"  coord_encoded: {coord_encoded.shape}")
    print(f"Output shape: {distances.shape}")
    print(f"Output range: [{distances.min().item():.4f}, {distances.max().item():.4f}]")

    # 测试不启用自定义初始化的版本
    print("\n=== Without Custom Initialization ===")
    metric_net_default = MetricNet(
        feat_dim=1024,
        coord_dim=128,
        init_weights=False  # 不启用自定义初始化
    )

    with torch.no_grad():
        bias = metric_net_default.coord_to_film.fc3.bias
        bias_reshaped = bias.view(3, 2, 512)
        gamma_bias = bias_reshaped[:, 0, :]
        beta_bias = bias_reshaped[:, 1, :]

        print(f"Gamma bias (default) - mean: {gamma_bias.mean():.4f}, std: {gamma_bias.std():.4f}")
        print(f"Beta bias (default) - mean: {beta_bias.mean():.4f}, std: {beta_bias.std():.4f}")