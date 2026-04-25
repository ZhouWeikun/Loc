import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Scalar-p GeM used as the current baseline implementation.
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class G2M(nn.Module):
    """
    G2M with scalar GeM exponents in both branches.
    """

    def __init__(self, in_channels=768, out_channels=768, rank=64, p=3.0, eps=1e-6):
        super().__init__()
        self.output_dim = out_channels
        self.main_gem = GeM(p=p, eps=eps)
        self.final_fc = nn.Linear(in_channels, out_channels)
        self.gca_gem = GeM(p=p, eps=eps)
        self.gca_mlp = nn.Sequential(
            nn.Linear(in_channels, rank),
            nn.GELU(),
            nn.Linear(rank, in_channels),
        )
        self.gca_sigmoid = nn.Sigmoid()

    def forward(self, x, normalize=True):
        main_feat = self.main_gem(x).flatten(1)
        attention_feat = self.gca_gem(x).flatten(1)
        attention_weights = self.gca_sigmoid(self.gca_mlp(attention_feat))
        calibrated_feat = main_feat * attention_weights
        descriptor = self.final_fc(calibrated_feat)
        return F.normalize(descriptor, p=2, dim=1) if normalize else descriptor
