r'''
    MLP backbone of convolutional occupancy networks
    https://github.com/autonomousvision/convolutional_occupancy_networks
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, norm_type='none'):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        # === 新增：根据 norm_type 参数决定是否创建归一化层 ===
        self.norm_type = norm_type
        if self.norm_type == 'layernorm':
            self.norm_layer = nn.LayerNorm(size_h)
        elif self.norm_type != 'none':
            raise ValueError(f"未知的归一化类型: {self.norm_type}")
        # =================================================

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        # actvn -> fc_0
        net = self.fc_0(self.actvn(x))
        # 在第一个全连接层之后、第二个激活函数之前插入归一化层, 这是最常见的插入位置
        if self.norm_type == 'layernorm':
            net = self.norm_layer(net)
        # actvn -> fc_1
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=5, output_dim=1, c_opteration='add', norm_type='none',
                 leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size,norm_type=norm_type) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, output_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.c_operator = torch.add if c_opteration == "add" else torch.mul


    def forward(self, p, c_plane, **kwargs):
        c = c_plane
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = self.c_operator(net,self.fc_c[i](c))
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalDecoderFiLM(nn.Module):
    '''
    一个采用分层特征调制 (FiLM) 的解码器。
    它包含一个主干 MLP 和一个条件 MLP，实现对等的深度特征处理。
    '''

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=5, output_dim=1,norm_type='none', leaky=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size

        # 1. 主干 MLP (处理查询 p)
        self.fc_p = nn.Linear(dim, hidden_size)
        self.main_blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size, norm_type=norm_type) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # 2. 条件 MLP (深度处理条件 c)
        self.cond_mlp = nn.ModuleList()
        # 条件 MLP 的第一层
        self.cond_mlp.append(nn.Linear(c_dim, hidden_size))
        # 条件 MLP 的中间层
        for _ in range(n_blocks - 1):
            self.cond_mlp.append(nn.Linear(hidden_size, hidden_size))

        # 3. 调制层 (从条件 MLP 的输出生成 gamma 和 beta)
        # 每个 block 都需要一个调制器
        self.modulators = nn.ModuleList([
            # 输出维度是 hidden_size * 2，一半给 gamma，一半给 beta
            nn.Linear(hidden_size, hidden_size * 2) for _ in range(n_blocks)
        ])

        self.actvn = F.leaky_relu if leaky else F.relu

    def forward(self, p, c, **kwargs):
        p = p.float()

        # === 1. 深度处理条件 c ===
        # cond_features 列表将存储条件 MLP 每个中间层的输出
        cond_features = []
        c_feat = c
        for i in range(self.n_blocks):
            c_feat = self.cond_mlp[i](c_feat)
            c_feat = self.actvn(c_feat)
            cond_features.append(c_feat)

        # === 2. 处理查询 p，并在每层进行调制 ===
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            # 从条件 MLP 的第 i 层输出生成 gamma 和 beta
            # cond_features[i] 是第 i 个 block 对应的条件特征
            mod = self.modulators[i](cond_features[i])
            # 将输出切分为 gamma 和 beta
            # gamma 的值通常最好在 1 附近，所以加上 1
            gamma = mod[..., :self.hidden_size] + 1
            beta = mod[..., self.hidden_size:]

            # 应用 FiLM 调制
            net = gamma * net + beta

            # 通过主干 MLP 的 ResNet 块
            net = self.main_blocks[i](net)

        # === 3. 输出最终结果 ===
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out

