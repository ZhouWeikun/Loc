import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter_block import AdapterBlock
from .vision_transformer import vit_base, vit_giant2, vit_large, vit_small


DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

DINOV2_ALIASES = {
    "dinov2": "dinov2_vitb14",
}

DINOV2_MODEL_CONFIGS = {
    "dinov2_vits14": {
        "builder": vit_small,
        "patch_size": 14,
        "img_size": 518,
        "init_values": 1.0,
        "ffn_layer": "mlp",
    },
    "dinov2_vitb14": {
        "builder": vit_base,
        "patch_size": 14,
        "img_size": 518,
        "init_values": 1.0,
        "ffn_layer": "mlp",
    },
    "dinov2_vitl14": {
        "builder": vit_large,
        "patch_size": 14,
        "img_size": 518,
        "init_values": 1.0,
        "ffn_layer": "mlp",
    },
    "dinov2_vitg14": {
        "builder": vit_giant2,
        "patch_size": 14,
        "img_size": 518,
        "init_values": 1.0,
        "ffn_layer": "swiglufused",
    },
}

DEFAULT_DINOV2_WEIGHTS_DIR = "/root/.cache/torch/hub/checkpoints"


def _resolve_model_name(model_name):
    resolved_model_name = DINOV2_ALIASES.get(model_name, model_name)
    if resolved_model_name not in DINOV2_MODEL_CONFIGS:
        raise AssertionError(f"Unknown model name {model_name}")
    return resolved_model_name


def _resolve_pretrained_path(model_name, pretrained_path="", weights_dir=DEFAULT_DINOV2_WEIGHTS_DIR):
    if pretrained_path:
        path = os.path.abspath(pretrained_path)
    else:
        path = os.path.join(weights_dir, f"{model_name}_pretrain.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DINOv2 pretrained weights not found: {path}")
    return path


def build_dinov2_model(model_name, pretrained=True, pretrained_path="", weights_dir=DEFAULT_DINOV2_WEIGHTS_DIR):
    resolved_model_name = _resolve_model_name(model_name)
    model_cfg = DINOV2_MODEL_CONFIGS[resolved_model_name]
    model = model_cfg["builder"](
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        init_values=model_cfg["init_values"],
        ffn_layer=model_cfg["ffn_layer"],
        block_chunks=0,
    )

    if pretrained:
        state_dict = torch.load(
            _resolve_pretrained_path(
                resolved_model_name,
                pretrained_path=pretrained_path,
                weights_dir=weights_dir,
            ),
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model


def _resolve_n_last_blocks_with_adapter(model, adapter_config):
    adapter_config = dict(adapter_config or {})
    if not adapter_config.get("enabled", False):
        return 0
    total_blocks = len(model.blocks)
    raw_value = adapter_config.get("n_last_blocks_with_adapter", total_blocks)
    try:
        n_last_blocks = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"adapter_config.n_last_blocks_with_adapter must be an integer, got {raw_value!r}"
        ) from exc
    if n_last_blocks < 0:
        raise ValueError(
            f"adapter_config.n_last_blocks_with_adapter must be >= 0, got {n_last_blocks}"
        )
    return min(n_last_blocks, total_blocks)


def _apply_adapters(model, adapter_config):
    adapter_config = dict(adapter_config or {})
    n_last_blocks_with_adapter = _resolve_n_last_blocks_with_adapter(model, adapter_config)
    if n_last_blocks_with_adapter <= 0:
        return model, 0
    if getattr(model, "chunked_blocks", False):
        raise NotImplementedError("AdapterBlock replacement currently expects block_chunks=0.")

    adapted_blocks = []
    first_adapter_idx = len(model.blocks) - n_last_blocks_with_adapter
    for block_idx, block in enumerate(model.blocks):
        if block_idx < first_adapter_idx:
            adapted_blocks.append(block)
        else:
            adapted_blocks.append(AdapterBlock(block, dim=model.embed_dim, adapter_config=adapter_config))
    model.blocks = nn.ModuleList(adapted_blocks)
    return model, n_last_blocks_with_adapter


class DINOv2(nn.Module):
    """
    Local DINOv2 wrapper backed by the vendored official implementation.

    Args:
        model_name: one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks: number of last transformer blocks to unfreeze.
        norm_layer: if True, apply final norm before returning features.
        return_token: if True and reshape_output=True, return (patch_features, cls_token).
        reshape_output: if True, return patch features as [B, C, H/patch, W/patch].
        pretrained: whether to load official pretrained weights.
        pretrained_path: optional explicit local checkpoint path.
        weights_dir: fallback directory for cached official checkpoints.
        local_ckpt_dir: kept for compatibility with the old hub-based wrapper; ignored now.
        adapter_config: reserved for the upcoming adapter implementation.
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        num_trainable_blocks=2,
        norm_layer=False,
        return_token=False,
        reshape_output=False,
        pretrained=True,
        pretrained_path="",
        weights_dir=DEFAULT_DINOV2_WEIGHTS_DIR,
        local_ckpt_dir=None,
        adapter_config=None,
    ):
        super().__init__()

        self.model_name = _resolve_model_name(model_name)
        self.model = build_dinov2_model(
            self.model_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            weights_dir=weights_dir,
        )
        self.num_channels = DINOV2_ARCHS[self.model_name]
        self.requested_num_trainable_blocks = int(num_trainable_blocks)
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.reshape_output = reshape_output
        self.adapter_config = dict(adapter_config or {})
        self.normalize_before_aggregator = bool(self.adapter_config.get("normalize_before_aggregator", False))
        self.model, self.n_last_blocks_with_adapter = _apply_adapters(self.model, self.adapter_config)
        if self.adapter_config.get("enabled", False):
            self.adapter_config["n_last_blocks_with_adapter"] = self.n_last_blocks_with_adapter
            if self.requested_num_trainable_blocks > 0:
                print(
                    f"[DINOv2] adapter enabled; ignoring num_trainable_blocks="
                    f"{self.requested_num_trainable_blocks} and freezing raw backbone weights."
                )
            self.num_trainable_blocks = 0
        else:
            self.num_trainable_blocks = self.requested_num_trainable_blocks
        self._configure_parameter_trainability()

    @property
    def patch_size(self):
        patch_size = getattr(self.model, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    def _configure_parameter_trainability(self):
        total_blocks = len(self.model.blocks)
        self.num_trainable_blocks = max(0, min(self.num_trainable_blocks, total_blocks))

        for param in self.model.parameters():
            param.requires_grad = False

        if self.adapter_config.get("enabled", False):
            for blk in self.model.blocks[-self.n_last_blocks_with_adapter:]:
                if isinstance(blk, AdapterBlock):
                    for param in blk.iter_adapter_parameters():
                        param.requires_grad = True
            return

        if self.num_trainable_blocks <= 0:
            return

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            for param in blk.parameters():
                param.requires_grad = True

        if self.norm_layer and hasattr(self.model, "norm"):
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def forward(self, x):
        bsz, _, height, width = x.shape
        n_trainable = self.num_trainable_blocks
        total_blocks = len(self.model.blocks)
        adapter_enabled = bool(self.adapter_config.get("enabled", False))
        n_adapter_blocks = int(getattr(self, "n_last_blocks_with_adapter", 0))

        with torch.no_grad():
            x = self.model.prepare_tokens_with_masks(x)
        x = x.detach()

        if adapter_enabled:
            prefix_blocks = self.model.blocks[:-n_adapter_blocks] if n_adapter_blocks > 0 else self.model.blocks
            adapter_blocks = self.model.blocks[-n_adapter_blocks:] if n_adapter_blocks > 0 else []

            with torch.no_grad():
                for blk in prefix_blocks:
                    x = blk(x)
            x = x.detach()

            for blk in adapter_blocks:
                x = blk(x)
            if self.norm_layer:
                x = self.model.norm(x)
        elif n_trainable <= 0:
            if not adapter_enabled:
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

        if self.normalize_before_aggregator:
            x = F.normalize(x, p=2, dim=-1)

        if self.reshape_output:
            token = x[:, 0]
            feat = x[:, 1:]
            patch_size = self.patch_size
            feat = feat.reshape((bsz, height // patch_size, width // patch_size, self.num_channels)).permute(0, 3, 1, 2)
            if self.return_token:
                return feat, token
            return feat
        return x

    def forward_features(self, x):
        return self.forward(x)
