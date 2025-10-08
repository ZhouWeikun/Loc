import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) a a single layer.
    This implementation is simpler and used as a building block for G2M.
    It does not include flatten or normalization.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # F.avg_pool2d is used to calculate the mean of the powered activations
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
        # The output shape is [B, C, 1, 1]


class G2M(nn.Module):
    """
    Implementation of G2M (Generalized Channel Attention for GeM) based on the SuperPlace paper.
    This module takes a feature map from a VFM (e.g., DINOv2) and outputs a global descriptor.
    """

    def __init__(self, in_channels=768, out_channels=768, rank=64, p=3.0, eps=1e-6):
        """
        Args:
            in_channels (int): Number of channels in the input feature map (e.g., 768 for DINOv2-Base).
            out_channels (int): Dimension of the final output descriptor.
            rank (int): The rank for the low-rank MLP in the GCA module.
            p (float): Initial value for the GeM pooling parameter.
            eps (float): Epsilon for numerical stability in GeM.
        """
        super(G2M, self).__init__()

        # 1. Main Branch
        self.main_gem = GeM(p=p, eps=eps)
        self.final_fc = nn.Linear(in_channels, out_channels)

        # 2. GCA (Generalized Channel Attention) Branch
        self.gca_gem = GeM(p=p, eps=eps)
        # Low-rank MLP for GCA
        self.gca_mlp = nn.Sequential(
            nn.Linear(in_channels, rank),
            nn.GELU(),
            nn.Linear(rank, in_channels)
        )
        self.gca_sigmoid = nn.Sigmoid()

    def forward(self, x, normalize=True):
        """
        Args:
            x (torch.Tensor): The input feature map from patch tokens, with shape [B, C, H, W].

        Returns:
            torch.Tensor: The final global descriptor, with shape [B, out_channels].
        """
        # --- Main Branch ---
        # Pool the feature map
        main_feat = self.main_gem(x)  # Shape: [B, C, 1, 1]
        main_feat = main_feat.flatten(1)  # Shape: [B, C]

        # --- GCA Branch ---
        # Pool the feature map to get channel-wise statistics
        attention_feat = self.gca_gem(x)  # Shape: [B, C, 1, 1]
        attention_feat = attention_feat.flatten(1)  # Shape: [B, C]

        # Calculate channel attention weights
        attention_weights = self.gca_mlp(attention_feat)  # Shape: [B, C]
        attention_weights = self.gca_sigmoid(attention_weights)  # Shape: [B, C]

        # --- Fusion ---
        # Calibrate the main feature vector with attention weights
        calibrated_feat = main_feat * attention_weights

        # --- Final Projection and Normalization ---
        # Project to the final descriptor dimension
        descriptor = self.final_fc(calibrated_feat)

        # L2 Normalize the final descriptor
        descriptor = F.normalize(descriptor, p=2, dim=1) if normalize else descriptor

        return descriptor



# --- Example Usage ---
if __name__ == '__main__':
    # Example parameters for DINOv2-Base
    B = 4  # Batch size
    C = 768  # Number of channels
    H, W = 16, 16  # Example feature map size (e.g., for a 224x224 image with patch size 14)

    # Create a dummy input tensor
    dummy_feature_map = torch.randn(B, C, H, W)

    # Instantiate the G2M model
    # Using parameters from the paper for DINOv2-Base
    g2m_model = G2M(in_channels=C, out_channels=768, rank=64)

    # Forward pass
    output_descriptor = g2m_model(dummy_feature_map)

    # Print shapes to verify
    print(f"Input feature map shape: {dummy_feature_map.shape}")
    print(f"Output descriptor shape: {output_descriptor.shape}")

    # Check if the output is L2 normalized
    norms = torch.norm(output_descriptor, p=2, dim=1)
    print(f"Norms of output descriptors: \n{norms}")