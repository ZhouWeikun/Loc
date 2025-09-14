import torch
import torch.nn as nn
from collections import OrderedDict

# def create_mlp(dims_list, activation_fn=nn.ReLU):
#     """
#     根据给定的维度序列和激活函数类型，动态创建一个MLP模型。
#
#     :param dims_list: 一个包含各层维度的列表或元组。
#                       例如：[input_dim, hidden1_dim, hidden2_dim, output_dim]
#     :param activation_fn: 一个 torch.nn 中的激活函数类（注意：是类本身，不是实例）。
#                           默认为 nn.ReLU。
#     :return: 一个 nn.Sequential 构成的 MLP 模型。
#     """
#     if len(dims_list) < 2:
#         raise ValueError("维度序列至少需要包含输入和输出两个维度。")
#
#     layers = []
#     # 循环创建隐藏层
#     for i in range(len(dims_list) - 2):
#         layers.append((f'linear_{i + 1}', nn.Linear(dims_list[i], dims_list[i + 1])))
#         # 使用传入的 activation_fn 类来实例化激活层
#         layers.append((f'activation_{i + 1}', activation_fn()))
#
#     # 添加输出层
#     layers.append(('output_layer', nn.Linear(dims_list[-2], dims_list[-1])))
#
#     return nn.Sequential(OrderedDict(layers))

from typing import Optional
def create_mlp(dims_list, activation_fn=nn.ReLU, norm_type: Optional[str] = None):
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