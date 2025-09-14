import torch
import torch.nn as nn
import math # 仅用于可能的维度计算展示，本代码中未使用

class SpatialSoftmaxPyTorch(nn.Module):
    """
    PyTorch implementation of the SpatialSoftmax module.

    This module applies a spatial softmax attention mechanism, where attention weights
    are computed across spatial dimensions for multiple heads, and these weights
    modulate a value projection. This is different from standard self-attention
    which computes pairwise interactions.

    Args:
        input_channels (int): Number of channels in the input tensor.
        channels (int): Total number of output channels for the value projection.
                        This will also be the number of output channels of this module.
                        Must be divisible by `heads`.
        heads (int): Number of attention heads.
    """
    def __init__(self, input_channels: int, channels: int, heads: int):
        super().__init__()

        if channels % heads != 0:
            raise ValueError(f"`channels` ({channels}) must be divisible by `heads` ({heads}).")

        self.input_channels = input_channels
        self.channels = channels  # Corresponds to c2 in (h c2) from einx.dot context
        self.heads = heads
        self.per_head_channels = channels // heads # Corresponds to c in (h c)

        # Assuming Norm() in Flax is equivalent to LayerNorm applied on the last dimension (features/channels)
        # PyTorch's LayerNorm takes the shape of the dimension(s) to normalize.
        # If input is (B, ..., C_in), LayerNorm(C_in) normalizes the last dimension.
        self.norm = nn.LayerNorm(input_channels)

        # Linear layer to compute attention logits
        # Original Flax/einx: attn = einn.Linear("... [c1->h]", h=self.heads, bias=False)(x)
        # Here, c1 is input_channels, and h (output features for this linear layer) is self.heads.
        self.attn_fc = nn.Linear(input_channels, self.heads, bias=False)

        # Linear layer to compute the value projection
        # Original Flax/einx: value = einn.Linear("... [c1->c2]", c2=self.channels)(x)
        # Here, c1 is input_channels, and c2 (output features) is self.channels.
        self.value_fc = nn.Linear(input_channels, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialSoftmax module.

        Args:
            x (torch.Tensor): Input tensor, assumed to be of shape
                              (batch_size, *spatial_dims, input_channels).
                              For example, for a 2D image-like input: (B, H, W, C_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, channels).
        """
        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected the last dimension of input tensor x to be `input_channels` ({self.input_channels}), "
                f"but got {x.shape[-1]}."
            )

        batch_size = x.shape[0]
        # Capture all spatial dimensions (e.g., H, W for images)
        # These are the dimensions represented by "..." in the einx strings.
        spatial_dims = x.shape[1:-1]
        num_spatial_elements = x.shape[1:-1].numel() # Product of spatial dimensions

        # 1. Normalization
        # Applies LayerNorm on the last dimension (input_channels)
        x_norm = self.norm(x)
        # Shape: (batch_size, *spatial_dims, input_channels)

        # 2. Compute Attention Logits
        # self.attn_fc maps (..., input_channels) -> (..., heads)
        attn_logits = self.attn_fc(x_norm)
        # Shape: (batch_size, *spatial_dims, heads)

        # 3. Compute Softmax for Attention Weights
        # Original Flax/einx: attn = einx.softmax("b [...] h", attn)
        # This means softmax is applied over the "[...]" (spatial) dimensions,
        # for each batch element 'b' and each head 'h' independently.
        # To achieve this, we reshape so that all spatial dimensions are flattened into one,
        # apply softmax along that dimension, and then reshape back.

        original_attn_shape = attn_logits.shape # (batch_size, *spatial_dims, heads)
        # Reshape for softmax: (batch_size, num_spatial_elements, heads)
        attn_logits_reshaped = attn_logits.reshape(batch_size, num_spatial_elements, self.heads)

        # Apply softmax along the flattened spatial dimension (dim=1)
        # For each head, the weights across all spatial locations will sum to 1.
        attn_weights_reshaped = torch.softmax(attn_logits_reshaped, dim=1)

        # Reshape back to the original spatial dimensions
        # Shape: (batch_size, *spatial_dims, heads)
        attn_weights = attn_weights_reshaped.reshape(original_attn_shape)

        # 4. Compute Value Projection
        # self.value_fc maps (..., input_channels) -> (..., self.channels)
        # where self.channels = self.heads * self.per_head_channels
        value = self.value_fc(x_norm)
        # Shape: (batch_size, *spatial_dims, self.channels)

        # 5. Apply Attention Weights to Value (mimicking einx.dot("b ... (h c), b ... h -> b (h c)", value, attn))
        #   - `value` has effective logical shape `b ... h c` (where `self.channels` is `h*c`)
        #   - `attn_weights` has shape `b ... h`
        #   - Output should have shape `b ... (h c)` (i.e., `b ... self.channels`)

        # Reshape value to explicitly expose head and per_head_channels dimensions:
        # Shape: (batch_size, *spatial_dims, self.heads, self.per_head_channels)
        value_reshaped = value.reshape(batch_size, *spatial_dims, self.heads, self.per_head_channels)

        # Expand attn_weights to be broadcastable with value_reshaped:
        # (batch_size, *spatial_dims, self.heads) -> (batch_size, *spatial_dims, self.heads, 1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)

        # Element-wise multiplication.
        # attn_weights_expanded will be broadcasted across the self.per_head_channels dimension.
        # (B, ..., H, W, heads, per_head_channels) * (B, ..., H, W, heads, 1)
        # -> (B, ..., H, W, heads, per_head_channels)
        weighted_value = value_reshaped * attn_weights_expanded
        # Shape: (batch_size, *spatial_dims, self.heads, self.per_head_channels)

        # Reshape back to combine heads and per_head_channels into the single `self.channels` dimension:
        # Shape: (batch_size, *spatial_dims, self.channels)
        output = weighted_value.reshape(batch_size, *spatial_dims, self.channels)

        return output