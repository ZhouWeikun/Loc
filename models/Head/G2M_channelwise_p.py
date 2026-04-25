import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Channel-wise GeM with one learnable exponent per channel.

    This follows the paper notation more closely, where the pooling exponent can
    vary across channels.
    """

    def __init__(self, channels, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(int(channels)) * float(p))
        self.eps = float(eps)

    def forward(self, x):
        p = self.p.view(1, -1, 1, 1).clamp_min(self.eps)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class G2M(nn.Module):
    """
    G2M with channel-wise GeM exponents in both branches.
    """

    def __init__(self, in_channels=768, out_channels=768, rank=64, p=3.0, eps=1e-6):
        super().__init__()
        self.output_dim = int(out_channels)
        self.main_gem = GeM(channels=in_channels, p=p, eps=eps)
        self.final_fc = nn.Linear(in_channels, out_channels)
        self.gca_gem = GeM(channels=in_channels, p=p, eps=eps)
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
