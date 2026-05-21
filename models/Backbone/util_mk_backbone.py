import torch
import torch.nn as nn


def make_backbone(backbone_name, imgsize2net=224, backbone_config=None, adapter_config=None):
    return Backbone(
        backbone_name,
        imgsize2net,
        backbone_config=backbone_config,
        adapter_config=adapter_config,
    )


class Backbone(nn.Module):
    """Minimal visual backbone wrapper supporting DINOv2 and DINOv3 only."""

    def __init__(self, backbone_name, imgsize2net=224, backbone_config=None, adapter_config=None):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.imgsize2net = (imgsize2net, imgsize2net)
        self.backbone_config = dict(backbone_config or {})
        self.adapter_config = dict(adapter_config or {})
        self.backbone, self.output_channel = self.init_backbone()

    def init_backbone(self):
        backbone = self.backbone_name
        backbone_l = backbone.lower()

        if "dino" in backbone_l and "v2" in backbone_l:
            from models.Backbone.dinov2 import DINOv2, DINOV2_ALIASES, DINOV2_ARCHS

            resolved_backbone = DINOV2_ALIASES.get(backbone, backbone)
            backbone_model = DINOv2(
                model_name=resolved_backbone,
                adapter_config=self.adapter_config,
                **self.backbone_config,
            )
            return backbone_model, DINOV2_ARCHS[resolved_backbone]

        if "dino" in backbone_l and "v3" in backbone_l:
            from models.Backbone.dinov3.vision_transformer import vit_large

            backbone_model = vit_large(
                patch_size=16,
                layerscale_init=1.0e-05,
                n_storage_tokens=4,
                mask_k_bias=True,
                untie_global_and_local_cls_norm=True,
                qkv_bias=True,
                drop_path_rate=0.0,
                norm_layer="layernormbf16",
                ffn_layer="mlp",
                ffn_bias=True,
                proj_bias=True,
            )
            local_weights_path = self.backbone_config.get("weights_path", "")
            if not local_weights_path:
                raise ValueError("DINOv3 requires backbone_config.weights_path.")
            checkpoint = torch.load(local_weights_path, map_location="cpu")
            backbone_model.load_state_dict(checkpoint, strict=False)
            return backbone_model, 1024

        raise NameError(f"{backbone} is not supported by the minimal backbone factory.")

    def forward(self, image):
        features = self.backbone.forward_features(image)
        if "dino" in self.backbone_name.lower() and "v3" in self.backbone_name.lower():
            x_norm_clstoken = features["x_norm_clstoken"]
            x_norm_patchtokens = features["x_norm_patchtokens"]
            features = torch.concatenate([x_norm_clstoken.unsqueeze(1), x_norm_patchtokens], dim=1)
        return features
