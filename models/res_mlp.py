import torch
import torch.nn as nn
from typing import Dict, Type,Optional


class ResnetBlockFC(nn.Module):
    """全连接残差块

    Args:
        input_dim: 输入维度
        output_dim: 输出维度（默认与输入相同）
        hidden_dim: 隐藏层维度（默认为输入和输出的最小值）
        norm_type: 归一化类型 ('none', 'layernorm')
        activation_type: 激活函数类型 ('relu', 'leaky_relu', 'gelu', 'silu', 'none')
        init_type: 权重初始化类型 ('default', 'resnet_zero_init', 'kaiming_normal', 'xavier_uniform')
    """

    def __init__(self, input_dim: int,
                 output_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None,
                 norm_type: str = 'none',
                 activation_type: str = 'relu',
                 init_type: str = 'resnet_zero_init'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = min(input_dim, output_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_type = init_type

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.activation = self._get_activation_layer(activation_type)

        if norm_type == 'layernorm':
            self.norm_layer = nn.LayerNorm(hidden_dim)
        elif norm_type != 'none':
            raise ValueError(f"Unknown norm type: {norm_type}")
        self.norm_type = norm_type

        self.shortcut = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None
        self._initialize_weights()

    def _get_activation_layer(self, activation_type: str) -> nn.Module:
        activation_options: Dict[str, Type[nn.Module]] = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'none': nn.Identity
        }
        if activation_type.lower() not in activation_options:
            raise ValueError(f"Unknown activation type: {activation_type}")
        return activation_options[activation_type.lower()]()

    def _initialize_weights(self):
        if self.init_type == 'default':
            return
        if self.init_type == 'resnet_zero_init':
            nn.init.zeros_(self.fc_2.weight)
        elif self.init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(self.fc_1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.fc_1.bias)
            nn.init.kaiming_normal_(self.fc_2.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.fc_2.bias)
        elif self.init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(self.fc_1.weight)
            nn.init.zeros_(self.fc_1.bias)
            nn.init.xavier_uniform_(self.fc_2.weight)
            nn.init.zeros_(self.fc_2.bias)
        else:
            raise ValueError(f"Unknown initialization type: {self.init_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（预激活残差块）"""
        # 恒等路径/捷径
        if self.shortcut is not None:
            identity = self.shortcut(x)
        else:
            identity = x

        # 残差路径
        residual = self.activation(x)
        residual = self.fc_1(residual)
        if self.norm_type == 'layernorm':
            residual = self.norm_layer(residual)
        residual = self.activation(residual)
        residual = self.fc_2(residual)

        return identity + residual

    # def forward(self, x: torch.Tensor) -> torch.Tensor:  # 后激活版本
    #     # --- 1. 计算恒等路径/捷径 (Identity/Shortcut Path) ---
    #     if self.shortcut is not None:
    #         identity = self.shortcut(x)
    #     else:
    #         identity = x
    #
    #     # --- 2. 计算残差路径 (Residual Path) ---
    #     residual = self.fc_1(x)
    #     if self.norm_type == 'layernorm':
    #         residual = self.norm_layer(residual)
    #     residual = self.activation(residual)
    #     residual = self.fc_2(residual)
    #
    #     return identity + residual

