import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
        self,
        dim,
        bottleneck_dim,
        dropout=0.0,
        init="zero_up",
        skip_connect=False,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.reset_parameters(init)

    def reset_parameters(self, init):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.down_proj.bias)
        if init == "zero_up":
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
        elif init == "small_random":
            nn.init.uniform_(self.up_proj.weight, -1.0e-4, 1.0e-4)
            nn.init.zeros_(self.up_proj.bias)
        else:
            raise ValueError(f"Unknown adapter init: {init}")

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        if self.skip_connect:
            x = x + residual
        return x


class AdapterBlock(nn.Module):
    def __init__(self, base_block, dim, adapter_config):
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.ls1 = base_block.ls1
        self.drop_path1 = base_block.drop_path1
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp
        self.ls2 = base_block.ls2
        self.drop_path2 = base_block.drop_path2
        self.sample_drop_ratio = getattr(base_block, "sample_drop_ratio", 0.0)

        bottleneck_dim = int(adapter_config.get("bottleneck_dim", 64))
        dropout = float(adapter_config.get("dropout", 0.0))
        init = str(adapter_config.get("init", "zero_up"))
        self.parallel_scale = float(adapter_config.get("scale", 0.2))

        self.attn_adapter = None
        if bool(adapter_config.get("attn_adapter", True)):
            self.attn_adapter = Adapter(
                dim=dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                init=init,
                skip_connect=False,
            )

        self.ffn_adapter = None
        if bool(adapter_config.get("ffn_parallel_adapter", True)):
            self.ffn_adapter = Adapter(
                dim=dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                init=init,
                skip_connect=False,
            )

    def iter_adapter_parameters(self):
        if self.attn_adapter is not None:
            yield from self.attn_adapter.parameters()
        if self.ffn_adapter is not None:
            yield from self.ffn_adapter.parameters()

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        if self.attn_adapter is not None:
            attn_out = self.attn_adapter(attn_out)
        x = x + self.drop_path1(self.ls1(attn_out))

        normed = self.norm2(x)
        ffn_out = self.mlp(normed)
        if self.ffn_adapter is not None:
            ffn_out = ffn_out + self.parallel_scale * self.ffn_adapter(normed)
        x = x + self.drop_path2(self.ls2(ffn_out))
        return x
