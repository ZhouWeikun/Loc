import torch
import torch.nn as nn

from models.res_mlp import ResnetBlockFC
# from .metric_net import BranchMLP,CoordToFiLM

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

class DualStreamMetricNet(nn.Module):
    """
    【激进修改版】双流直连架构 (No Interaction Terms)

    逻辑：
    Distance = MLP( Concatenate( Branch(Query), FiLM_Modulated_Branch(Ref) ) )

    特点：
    1. 移除 mul/sub/sq 交互项，减少对噪声的敏感度。
    2. 网络必须自己学会 "Compare" 操作。
    3. 合法输入：只使用 ref 的坐标进行调制，绝不泄露 query 坐标。
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

        self.output_activation = output_activation

        # ============ 1. 双流分支 (Query & Reference) ============
        # Query分支: 处理无人机视觉特征
        self.branch_query = BranchMLP(
            feat_dim, branch_hidden_dim, branch_output_dim, dropout
        )
        # Ref分支: 处理Grid特征
        self.branch_ref = BranchMLP(
            feat_dim, branch_hidden_dim, branch_output_dim, dropout
        )

        # ============ 2. 坐标调制 (仅针对Ref) ============
        # 我们只调制 Ref 分支，因为 Ref 特征是空间敏感的
        # Query 特征是“待匹配的目标”，可以保持语义不变
        self.coord_to_film = CoordToFiLM(
            coord_dim=coord_dim,
            hidden_dim=256,
            output_dim=branch_output_dim,
            num_branches=1  # 只需要调制 Ref 这一路
        )

        # ============ 3. 残差提取块 ============
        self.resblock_query = ResnetBlockFC(
            input_dim=branch_output_dim,
            output_dim=resblock_output_dim,
            hidden_dim=resblock_hidden_dim,
            norm_type='none',
            activation_type='relu',
            init_type='resnet_zero_init'
        )
        self.resblock_ref = ResnetBlockFC(
            input_dim=branch_output_dim,
            output_dim=resblock_output_dim,
            hidden_dim=resblock_hidden_dim,
            norm_type='none',
            activation_type='relu',
            init_type='resnet_zero_init'
        )

        # ============ 4. 融合层 (关键) ============
        # 输入: [Query_Feat, Ref_Feat]
        # 网络需要在这里学会：如果 Q 和 R 很像，输出 0；如果不像，输出距离
        self.fusion_input_dim = resblock_output_dim * 2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 1)  # 输出标量距离
        )

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        # 通用初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # FiLM 初始化 (Gamma=1, Beta=0)
        nn.init.normal_(self.coord_to_film.fc3.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            bias = self.coord_to_film.fc3.bias
            # Reshape to [1 branch, 2 params, dim]
            bias_reshaped = bias.view(1, 2, -1)
            bias_reshaped[:, 0, :] = 1.0  # Gamma
            bias_reshaped[:, 1, :] = 0.0  # Beta
            bias.copy_(bias_reshaped.view(-1))

        # Fusion层最后一层初始化为接近0，防止初始Loss过大
        nn.init.normal_(self.fusion_mlp[-1].weight, mean=0.0, std=0.001)

    def forward(self,
                feat_query: torch.Tensor,
                feat_ref: torch.Tensor,
                coord_ref_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_query: [B, N, C] 无人机特征
            feat_ref:   [B, N, C] Grid特征 (包含潜在的噪声)
            coord_ref_encoded: [B, N, D] 当前评估位置的坐标编码
        """

        # 1. 维度对齐
        input_is_2d = (feat_ref.dim() == 2)
        if input_is_2d:
            feat_query = feat_query.unsqueeze(0)
            feat_ref = feat_ref.unsqueeze(0)
            coord_ref_encoded = coord_ref_encoded.unsqueeze(0)

        B, N, C = feat_ref.shape
        if feat_query.dim() == 2:
            feat_query = feat_query.unsqueeze(1).expand(B, N, C)

        # ============ Stream 1: Query (Visual) ============
        # 纯粹的视觉特征处理
        h_q = self.branch_query(feat_query)  # [B, N, 512]
        h_q_final = self.resblock_query(h_q)  # [B, N, 256]

        # ============ Stream 2: Reference (Grid) ============
        # 这一路包含了位置信息注入
        h_r = self.branch_ref(feat_ref)  # [B, N, 512]

        # FiLM 调制: 用坐标去"校准"Grid特征
        # 这是一个合法的操作：告诉网络"这是位置X的特征"
        gamma, beta = self.coord_to_film(coord_ref_encoded)  # gamma: [B, N, 1, 512], beta: [B, N, 1, 512]
        gamma = gamma[:, :, 0, :]  # [B, N, 512]
        beta = beta[:, :, 0, :]   # [B, N, 512]

        h_r_mod = gamma * h_r + beta  # [B, N, 512]
        h_r_final = self.resblock_ref(h_r_mod)  # [B, N, 256]

        # ============ Fusion: Concatenation ============
        # [Query, Ref] -> 拼接
        # 网络必须自己学会：当 Query ≈ Ref 时输出 0，否则输出 >0
        h_concat = torch.cat([h_q_final, h_r_final], dim=-1)  # [B, N, 512]

        distance = self.fusion_mlp(h_concat)

        if self.output_activation is not None:
            distance = self.output_activation(distance)

        distance = distance.squeeze(-1)
        if input_is_2d:
            distance = distance.squeeze(0)

        return distance
