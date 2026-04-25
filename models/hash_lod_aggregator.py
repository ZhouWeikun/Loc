import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleLodAggregator(nn.Module):
    """Scale-conditioned aggregation for concatenated HashGrid LOD features."""

    def __init__(
        self,
        num_lods=4,
        per_lod_dim=1024,
        output_dim=1024,
        coord_source='scale',
        scale_source='norm_log_scale',
        hidden_dim=64,
        temperature=1.0,
    ):
        super().__init__()
        self.num_lods = int(num_lods)
        self.per_lod_dim = int(per_lod_dim)
        self.output_dim = int(output_dim)
        self.coord_source = str(coord_source)
        self.scale_source = str(scale_source)
        self.temperature = float(temperature)
        self.input_dim = self.num_lods * self.per_lod_dim
        self.last_lod_weights = None

        if self.coord_source != 'scale':
            raise ValueError(f"Unsupported coord_source: {self.coord_source}")
        if self.scale_source not in {'norm_log_scale', 'last_dim'}:
            raise ValueError(f"Unsupported scale_source: {self.scale_source}")
        if self.num_lods <= 0:
            raise ValueError(f"num_lods must be positive, got {self.num_lods}")
        if self.per_lod_dim <= 0:
            raise ValueError(f"per_lod_dim must be positive, got {self.per_lod_dim}")
        if self.output_dim != self.per_lod_dim:
            raise ValueError(
                "ScaleLodAggregator currently uses weighted-sum aggregation, "
                f"so output_dim must equal per_lod_dim ({self.per_lod_dim}), got {self.output_dim}"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

        self.weight_mlp = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.num_lods),
        )
        self._init_uniform_weights()

    def _init_uniform_weights(self):
        last = self.weight_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _select_scale(self, coords_6d):
        if self.scale_source == 'norm_log_scale':
            if coords_6d.shape[-1] < 5:
                raise ValueError(
                    f"scale_source=norm_log_scale requires coords dim >= 5, got {coords_6d.shape[-1]}"
                )
            return coords_6d[..., 4:5]
        return coords_6d[..., -1:]

    def forward(self, feats_cat, coords_6d, return_weights=False):
        if feats_cat.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected HashGrid feature dim {self.input_dim} "
                f"(num_lods={self.num_lods}, per_lod_dim={self.per_lod_dim}), "
                f"got {feats_cat.shape[-1]}"
            )
        if coords_6d.shape[:-1] != feats_cat.shape[:-1]:
            raise ValueError(
                f"coords_6d leading shape {coords_6d.shape[:-1]} must match "
                f"feats_cat leading shape {feats_cat.shape[:-1]}"
            )

        feats = feats_cat.float().view(*feats_cat.shape[:-1], self.num_lods, self.per_lod_dim)
        scale = self._select_scale(coords_6d).float()
        logits = self.weight_mlp(scale) / self.temperature
        weights = F.softmax(logits, dim=-1)
        aggregated = (feats * weights.unsqueeze(-1)).sum(dim=-2)
        self.last_lod_weights = weights.detach()
        if return_weights:
            return aggregated, weights
        return aggregated
