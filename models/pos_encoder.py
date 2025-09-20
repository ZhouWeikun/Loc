import torch
import torch.nn as nn

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoder(nn.Module):
    """
    将坐标映射到高维特征的位置编码器。
    继承自 nn.Module，可以作为一个标准的 PyTorch 层来使用。
    """

    def __init__(self, input_dims, multires, include_input=True, log_sampling=True):
        """
        一次性初始化所有参数和函数。
        :param input_dims: 输入坐标的维度 (例如 2D 为 2, 3D 为 3)。
        :param multires: 多分辨率的级别，决定了频率的数量和范围。
        :param include_input: 是否在最终输出中包含原始输入坐标。
        :param log_sampling: 是否在对数空间中对频率进行采样。
        """
        super(PositionalEncoder, self).__init__()

        self.kwargs = {
            'include_input': include_input,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': log_sampling,
            'periodic_fns': [torch.sin, torch.cos],
        }

        self.embed_fns = []
        self.out_dim = 0

        # --- 这部分逻辑直接从原来的 create_embedding_fn 移入 ---
        if self.kwargs['include_input']:
            self.embed_fns.append(lambda x: x)
            self.out_dim += self.kwargs['input_dims']

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += self.kwargs['input_dims']

    def forward(self, inputs):
        """
        定义前向传播，当调用 encoder(inputs) 时会自动执行。
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def encode_4d_coords(coords, rc_encoder, rot_endcoder, scale_encoder):
    if len(coords.shape) == 3:
        b, n, _ = coords.shape
        coords = coords.reshape(-1, 4)
    else:
        b = 0
    rcs_encoded = rc_encoder(coords[:, :2])
    rots = torch.stack([torch.sin(coords[:, 2]), torch.cos(coords[:, 2])]).T
    rots_encoded = rot_endcoder(rots)
    scales_encoded = scale_encoder(coords[:, -1:])
    coords_encoded = torch.concatenate(
        [rcs_encoded, rots_encoded, scales_encoded], dim=-1)

    if b > 0:
        coords_encoded = coords_encoded.reshape(b, n, -1)

    return coords_encoded
