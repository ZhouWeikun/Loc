import torch
import torch.nn as nn
from collections import OrderedDict


from typing import Optional
def create_mlp(dims_list, activation_fn=nn.ReLU, norm_type: Optional[str] = None, dropout_p: Optional[float] = None):
    """
    根据给定的维度序列、激活函数和归一化类型，动态创建一个MLP模型。

    :param dims_list: 一个包含各层维度的列表或元组。
                      例如：[input_dim, hidden1_dim, hidden2_dim, output_dim]
    :param activation_fn: 一个 torch.nn 中的激活函数类（注意：是类本身，不是实例）。
                          默认为 nn.ReLU。
    :param norm_type: 归一化层的类型。可选值为 'layer', 'batch', 或 None。
                      'layer': 使用 Layer Normalization (层归一化)
                      'batch': 使用 Batch Normalization (批归一化)
                      None: 不使用任何归一化层。
                      默认为 None。
    :return: 一个 nn.Sequential 构成的 MLP 模型。
    """
    if len(dims_list) < 2:
        raise ValueError("维度序列至少需要包含输入和输出两个维度。")

    layers = []
    # 循环创建隐藏层
    for i in range(len(dims_list) - 2):
        # 1. 添加线性层
        layers.append((f'linear_{i + 1}', nn.Linear(dims_list[i], dims_list[i + 1])))

        # 2. (可选) 添加归一化层
        # 归一化层通常放在线性层之后、激活函数之前
        if norm_type == 'layer':
            # LayerNorm 需要归一化的特征维度，即当前层的输出维度
            layers.append((f'norm_{i + 1}', nn.LayerNorm(dims_list[i + 1])))
        elif norm_type == 'batch':
            # BatchNorm1d 也需要特征维度
            layers.append((f'norm_{i + 1}', nn.BatchNorm1d(dims_list[i + 1])))

        # 3. 添加激活层
        layers.append((f'activation_{i + 1}', activation_fn()))

        # 4. (可选) 添加Dropout层
        if dropout_p is not None and dropout_p > 0.0:
            layers.append((f'dropout_{i + 1}', nn.Dropout(p=dropout_p)))

    # 添加输出层
    # 通常输出层之前不加归一化和激活，除非有特殊需求
    layers.append(('output_layer', nn.Linear(dims_list[-2], dims_list[-1])))

    return nn.Sequential(OrderedDict(layers))


def init_weights(m, method='kaiming', nonlinearity='leaky_relu'):
    if isinstance(m, nn.Linear):
        if method == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        elif method == 'xavier':
            gain = nn.init.calculate_gain(nonlinearity)
            nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MultiMLP(nn.Module):
    def __init__(self, num_mlps: int, input_dim: int, mlp_hidden_dims: list,
                 mlp_activation_fn=nn.LeakyReLU, mlp_norm_type: Optional[str] = 'layer',
                 mlp_dropout_p: Optional[float] = None,  # <-- 新增参数
                 mlp_init_method: str = 'kaiming', mlp_init_nonlinearity: str = 'leaky_relu',
                 device='cpu'):
        """
        初始化 MultiMLP 模块，包含 N 个相同的 MLP，它们的输出经过 Softmax。

        Args:
            num_mlps (int): 要创建的相同 MLP 的数量 (N)。
            input_dim (int): 每个 MLP 的输入向量维度。
            mlp_hidden_dims (list): MLP 隐藏层的维度列表，例如 [128, 32, 8]。
                                    注意：这里的列表只包含隐藏层，不包含输入和输出层。
            mlp_activation_fn (class): 每个 MLP 隐藏层使用的激活函数类。默认为 nn.LeakyReLU。
            mlp_norm_type (str, optional): 每个 MLP 隐藏层使用的归一化类型。可选 'layer', 'batch', None。默认为 'layer'。
            mlp_init_method (str): MLP 权重初始化方法。'kaiming' 或 'xavier'。默认为 'kaiming'。
            mlp_init_nonlinearity (str): 初始化时计算增益使用的非线性函数字符串。默认为 'leaky_relu'。
            mlp_dropout_p (float, optional): 每个 MLP 隐藏层使用的Dropout概率。默认为 None。
            device (str): 模型所在的设备 ('cpu' 或 'cuda')。
        """
        super().__init__()
        self.num_mlps = num_mlps
        self.mlps = nn.ModuleList() # 使用 ModuleList 来存储多个 MLP

        # 为每个 MLP 构建完整的维度列表：输入 -> 隐藏层 -> 1 (单个输出神经元)
        full_mlp_dims = [input_dim] + mlp_hidden_dims + [1]

        print(f"Initializing {num_mlps} MLPs with dimensions: {full_mlp_dims}")
        print(f"Activation: {mlp_activation_fn.__name__}, Normalization: {mlp_norm_type}")
        if mlp_dropout_p is not None and mlp_dropout_p > 0.0:
            print(f"Dropout Probability: {mlp_dropout_p}")
        print(f"Weight Init: {mlp_init_method} with nonlinearity: {mlp_init_nonlinearity}")

        for i in range(self.num_mlps):
            # 使用 create_mlp 函数创建单个 MLP
            mlp = create_mlp(
                dims_list=full_mlp_dims,
                activation_fn=mlp_activation_fn,
                norm_type=mlp_norm_type,
                dropout_p = mlp_dropout_p
            )
            # 使用 init_weights 函数初始化 MLP 的权重
            mlp.apply(lambda m: init_weights(m, method=mlp_init_method, nonlinearity=mlp_init_nonlinearity))
            self.mlps.append(mlp.to(device))
        self.device = device

        # Softmax 层，作用于输出的 N 个值。dim=1 表示对每个样本的 N 个 MLP 输出进行 Softmax
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MultiMLP 的前向传播。

        Args:
            x (torch.Tensor): 输入向量，形状为 (batch_size, input_dim)。

        Returns:
            torch.Tensor: 经过 Softmax 后的输出概率，形状为 (batch_size, num_mlps)。
        """
        mlp_outputs = []

        # 让每个 MLP 处理相同的输入
        for mlp in self.mlps:
            output = mlp(x) # output 形状为 (batch_size, 1)
            mlp_outputs.append(output)

        # 将所有 MLP 的输出在新的维度上堆叠起来
        # 结果形状为 (batch_size, num_mlps)
        stacked_outputs = torch.cat(mlp_outputs, dim=1)

        # 应用 Softmax 得到概率分布
        # probabilities = self.softmax(stacked_outputs)

        return stacked_outputs