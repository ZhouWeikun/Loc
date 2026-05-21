import torch
import torch.nn as nn
import torch.nn.functional as F
from models.res_mlp import ResnetBlockFC


class SerialModulatorShallow(nn.Module):
    """基于外部条件的浅层串行调制器

    该模块通过外部条件特征（如平面特征、体素特征等）对输入特征进行逐层调制，
    适用于条件生成任务（如隐式神经表示、占据场预测等）。

    Args:
        input_dim: 输入特征的维度（如 3D 坐标为 3，也可以是其他特征）
        condition_dim: 条件特征的维度（如平面特征维度）
        hidden_dim: 隐藏层维度
        num_blocks: ResNet 块的数量
        output_dim: 输出维度（默认为 1，用于 SDF/occupancy）
        condition_operator: 条件特征的融合方式 ('add' 或 'mul')
        output_activation_type: 输出层前的激活函数类型 ('relu', 'leaky_relu', 'gelu', 'silu', 'none')
        encoder_init_type: 输入编码层的初始化类型 ('default', 'kaiming_normal', 'xavier_uniform', 'zero', 'small_random')
        output_init_type: 输出层的初始化类型 ('default', 'kaiming_normal', 'xavier_uniform', 'zero', 'small_random')
        condition_init_type: 条件投影层的初始化类型 ('default', 'kaiming_normal', 'xavier_uniform', 'zero', 'small_random')
        block_norm_type: ResNet 块内的归一化类型 ('none', 'layernorm')
        block_activation_type: ResNet 块内的激活函数类型 ('relu', 'leaky_relu', 'gelu', 'silu', 'none')
        block_init_type: ResNet 块的权重初始化类型 ('default', 'resnet_zero_init', 'kaiming_normal', 'xavier_uniform')
    """

    def __init__(self,
                 input_dim: int = 3,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 num_blocks: int = 5,
                 output_dim: int = 1,
                 condition_operator: str = 'add',
                 output_activation_type: str = 'relu',
                 encoder_init_type: str = 'kaiming_normal',
                 output_init_type: str = 'kaiming_normal',
                 condition_init_type: str = 'kaiming_normal',
                 block_norm_type: str = 'none',
                 block_activation_type: str = 'relu',
                 block_init_type: str = 'resnet_zero_init',
                 norm_output:bool = True,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.norm_output = norm_output

        # 输入特征的初始编码层
        self.input_encoder = nn.Linear(input_dim, hidden_dim)

        # 条件特征的线性投影层（每个 ResNet 块一个）
        if condition_dim != 0:
            self.condition_projectors = nn.ModuleList([
                nn.Linear(condition_dim, hidden_dim) for _ in range(num_blocks)
            ])

        # ResNet 残差块（使用统一的配置）
        self.resnet_blocks = nn.ModuleList([
            ResnetBlockFC(
                input_dim=hidden_dim,
                norm_type=block_norm_type,
                activation_type=block_activation_type,
                init_type=block_init_type
            ) for _ in range(num_blocks)
        ])

        # 最终输出层
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # 输出层前的激活函数
        self.output_activation = self._get_activation_function(output_activation_type)

        # 条件融合算子（加法或乘法）
        if condition_operator == 'add':
            self.condition_fusion = torch.add
        elif condition_operator == 'mul':
            self.condition_fusion = torch.mul
        else:
            raise ValueError(f"Unknown condition operator: {condition_operator}. Use 'add' or 'mul'.")

        # 初始化所有线性层
        self._initialize_linear_layer(self.input_encoder, encoder_init_type)
        self._initialize_linear_layer(self.output_head, output_init_type)
        if condition_dim != 0:
            for projector in self.condition_projectors:
                self._initialize_linear_layer(projector, condition_init_type)

    def _initialize_linear_layer(self, layer: nn.Linear, init_type: str):
        """统一的线性层初始化方法

        Args:
            layer: 要初始化的线性层
            init_type: 初始化类型
                - 'default': 使用 PyTorch 默认初始化（不做任何操作）
                - 'kaiming_normal': Kaiming 正态初始化（适合 ReLU）
                - 'kaiming_uniform': Kaiming 均匀初始化
                - 'xavier_normal': Xavier 正态初始化（适合 Sigmoid/Tanh）
                - 'xavier_uniform': Xavier 均匀初始化
                - 'zero': 零初始化（权重和偏置都为零）
                - 'small_random': 小范围随机初始化 (±0.0001)
        """
        if init_type == 'default':
            return
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'zero':
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'small_random':
            nn.init.uniform_(layer.weight, -0.0001, 0.0001)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}. "
                             f"Use 'default', 'kaiming_normal', 'kaiming_uniform', "
                             f"'xavier_normal', 'xavier_uniform', 'zero', or 'small_random'.")

    def _get_activation_function(self, activation_type: str):
        """获取激活函数

        Args:
            activation_type: 激活函数类型

        Returns:
            对应的激活函数
        """
        activation_type = activation_type.lower()

        if activation_type == 'relu':
            return F.relu
        elif activation_type == 'leaky_relu':
            return lambda x: F.leaky_relu(x, 0.2)
        elif activation_type == 'gelu':
            return F.gelu
        elif activation_type == 'silu':
            return F.silu
        elif activation_type == 'none':
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation type: {activation_type}. "
                             f"Use 'relu', 'leaky_relu', 'gelu', 'silu', or 'none'.")

    def forward(self, inputs: torch.Tensor, condition_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播

        Args:
            inputs: 输入特征，形状 (B, N, input_dim) 或 (B, input_dim)
            condition_features: 条件特征，形状 (B, N, condition_dim) 或 (B, condition_dim)

        Returns:
            输出预测值，形状 (B, N) 或 (B,)
        """
        inputs = inputs.float()

        # 编码输入特征
        hidden_features = self.input_encoder(inputs)

        # 逐块处理：条件调制 + ResNet 残差块
        for block_idx in range(self.num_blocks):
            # 如果有条件特征，进行调制
            if self.condition_dim != 0:
                condition_projected = self.condition_projectors[block_idx](condition_features)
                hidden_features = self.condition_fusion(hidden_features, condition_projected)

            # 通过 ResNet 块
            hidden_features = self.resnet_blocks[block_idx](hidden_features)

        # 最终激活并输出
        output = self.output_head(self.output_activation(hidden_features))

        # 压缩最后一维（如果输出维度为 1）
        output = output.squeeze(-1)
        # output = torch.nn.functional.normalize(output,dim=-1) if self.norm_output else output

        return output


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用默认配置（PyTorch 默认初始化）
    model_default = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='default',
        output_init_type='default',
        condition_init_type='default',
        block_norm_type='none',
        block_activation_type='relu',
        block_init_type='resnet_zero_init'
    )
    # 打印参数量
    total_params = sum(p.numel() for p in model_default.parameters())
    print(f"Total parameters: {total_params:,}")

    # 示例2: 使用 Kaiming 初始化（适合 ReLU 网络）
    model_kaiming = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='kaiming_normal',
        output_init_type='kaiming_normal',
        condition_init_type='kaiming_normal',
        block_norm_type='none',
        block_activation_type='relu',
        block_init_type='kaiming_normal'
    )

    # 示例3: 使用小初始化（适合 SDF 任务）
    model_sdf = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='kaiming_normal',
        output_init_type='small_random',  # 输出层使用小初始化
        condition_init_type='xavier_uniform',
        block_norm_type='layernorm',
        block_activation_type='relu',
        block_init_type='resnet_zero_init'
    )

    # 模拟输入
    batch_size = 4
    num_points = 1024
    inputs = torch.randn(batch_size, num_points, 3)
    condition_features = torch.randn(batch_size, num_points, 128)

    # 前向传播
    output1 = model_default(inputs, condition_features)
    output2 = model_kaiming(inputs, condition_features)
    output3 = model_sdf(inputs, condition_features)

    print(f"输入形状: inputs {inputs.shape}, condition {condition_features.shape}")
    print(f"默认模型输出形状: {output1.shape}")
    print(f"Kaiming 模型输出形状: {output2.shape}")
    print(f"SDF 模型输出形状: {output3.shape}")

    # 检查初始化效果
    print(f"\n输出层权重统计:")
    print(
        f"默认模型 - mean: {model_default.output_head.weight.mean():.6f}, std: {model_default.output_head.weight.std():.6f}")
    print(
        f"Kaiming 模型 - mean: {model_kaiming.output_head.weight.mean():.6f}, std: {model_kaiming.output_head.weight.std():.6f}")
    print(f"SDF 模型 - mean: {model_sdf.output_head.weight.mean():.6f}, std: {model_sdf.output_head.weight.std():.6f}")