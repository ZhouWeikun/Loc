import torch
import torch.nn as nn
import os

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}
DINOV2_ALIASES = {
    'dinov2': 'dinov2_vitb14',
}


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False,
            reshape_output=False, #new added
            local_ckpt_dir='/root/.cache/torch/hub/facebookresearch_dinov2_main', #new added
    ):
        super().__init__()

        resolved_model_name = DINOV2_ALIASES.get(model_name, model_name)
        assert resolved_model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        if local_ckpt_dir and os.path.exists(local_ckpt_dir):
            self.model = torch.hub.load(local_ckpt_dir, resolved_model_name, source='local')
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', resolved_model_name)
        self.model_name = resolved_model_name
        self.num_channels = DINOV2_ARCHS[resolved_model_name]
        self.num_trainable_blocks = int(num_trainable_blocks)
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.reshape_output = reshape_output
        self._configure_parameter_trainability()

    def _configure_parameter_trainability(self):
        total_blocks = len(self.model.blocks)
        self.num_trainable_blocks = max(0, min(self.num_trainable_blocks, total_blocks))

        for param in self.model.parameters():
            param.requires_grad = False

        if self.num_trainable_blocks <= 0:
            return

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            for param in blk.parameters():
                param.requires_grad = True

        if self.norm_layer and hasattr(self.model, 'norm'):
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape
        n_trainable = self.num_trainable_blocks
        total_blocks = len(self.model.blocks)

        # Patch embedding / positional token preparation stay frozen to match the
        # intended "only last N transformer blocks are trainable" behavior.
        with torch.no_grad():
            x = self.model.prepare_tokens_with_masks(x)
        x = x.detach()

        if n_trainable <= 0:
            with torch.no_grad():
                for blk in self.model.blocks:
                    x = blk(x)
                if self.norm_layer:
                    x = self.model.norm(x)
            x = x.detach()
        elif n_trainable >= total_blocks:
            for blk in self.model.blocks:
                x = blk(x)
            if self.norm_layer:
                x = self.model.norm(x)
        else:
            with torch.no_grad():
                for blk in self.model.blocks[:-n_trainable]:
                    x = blk(x)
            x = x.detach()
            for blk in self.model.blocks[-n_trainable:]:
                x = blk(x)
            if self.norm_layer:
                x = self.model.norm(x)

        if self.reshape_output:
            t = x[:, 0]
            f = x[:, 1:]

            # Reshape to (B, C, H, W)
            f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

            if self.return_token:
                return f, t
            return f
        else:
            return x

    def forward_features(self,x):
        return self.forward(x)
