import torch
import torch.nn as nn
import torch.nn.functional as F
from .ocn_mlp import ResnetBlockFC

class SerialModulatorDeep(nn.Module):
    '''
    一个采用逐层串行调制的解码器。
    主干网络在处理主信号 s 的每一层中，都会被一个与之一一对应的条件特征 c 所调制。
    条件特征 c 本身也经过一个并行的深度 MLP 逐层生成。
    '''

    def __init__(self, s_dim=3, c_dim=128, hidden_size=256, n_blocks=5, output_dim=1,c_operation='add', norm_type='none', leaky=False):  # <-- 使用 c_operation
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size

        # 1. 主干 MLP (处理查询 p)
        self.fc_s = nn.Linear(s_dim, hidden_size)
        self.main_blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size, norm_type=norm_type) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # 2. 条件 MLP (深度处理条件 c) - 保持不变
        self.cond_mlp = nn.ModuleList()
        self.cond_mlp.append(nn.Linear(c_dim, hidden_size))
        for _ in range(n_blocks - 1):
            self.cond_mlp.append(nn.Linear(hidden_size, hidden_size))

        # 激活函数
        self.actvn = F.leaky_relu if leaky else F.relu

        # 定义调制操作符
        self.c_operator = torch.add if c_operation == "add" else torch.mul

    def forward(self, s, c, **kwargs):
        s = s.float()

        # === 1. 深度处理条件 c (与 FiLM 版本相同) ===
        cond_features = []
        c_feat = c
        for i in range(self.n_blocks):
            c_feat = self.cond_mlp[i](c_feat)
            c_feat = self.actvn(c_feat)
            cond_features.append(c_feat)

        # === 2. 处理输入信号 s (signal)，并在每层进行调制 ===
        net = self.fc_s(s)
        for i in range(self.n_blocks):
            net = self.c_operator(net, cond_features[i])
            # 通过主干 MLP 的 ResNet 块
            net = self.main_blocks[i](net)

        # === 3. 输出最终结果 ===
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out