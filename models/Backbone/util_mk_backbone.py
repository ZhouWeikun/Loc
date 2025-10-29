# import optparse
# import os
import torch.nn as nn
# import timm
from .RKNet import RKNet
# from .cvt import get_cvt_models
import torch

def make_backbone(backbone_name, imgsize2net=224):
    """
    创建backbone模型

    Args:
        backbone_name: str, backbone名称 (e.g., "ViTB-224", "dinov2", "dinov3")
        imgsize2net: int, 输入图像尺寸，默认224

    Returns:
        Backbone: backbone模型实例
    """
    backbone_model = Backbone(backbone_name, imgsize2net)
    return backbone_model


class Backbone(nn.Module):
    def __init__(self, backbone_name, imgsize2net=224):
        """
        Args:
            backbone_name: str, backbone名称
            imgsize2net: int, 输入图像尺寸
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.imgsize2net = (imgsize2net, imgsize2net)
        self.backbone, self.output_channel = self.init_backbone()

    def init_backbone(self):
        backbone = self.backbone_name
        if backbone=="resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="RKNet":
            backbone_model = RKNet()
            output_channel = 2048
        elif backbone=="senet":
            backbone_model = timm.create_model('legacy_seresnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="ViTS-224":
            backbone_model = timm.create_model("vit_small_patch16_224", pretrained=True, imgsize2net=self.imgsize2net)
            output_channel = 384
        elif backbone=="ViTS-384":
            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True)
            output_channel = 384
        elif backbone=="DeitS-224":
            backbone_model = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="DeitB-224":
            backbone_model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="Pvtv2b2":
            backbone_model = timm.create_model("pvt_v2_b2", pretrained=True)
            output_channel = 512
        elif ('vitb'in backbone.lower()) and  ('224'in backbone):
            output_channel = 768
            # using the weights from official vit-b, offline
            model_name = 'vit_base_patch16_224'  # 请确认这个名称是否正确
            local_weights_path = "/home/data/zwk/ckpt_vitb224/vit_base_patch16_224.augreg2_in21k_ft_in1k_ckpt/pytorch_model.bin"  # <--- 修改成你的实际路径
            backbone_model = timm.create_model(
                model_name,
                pretrained=False,  # 设置为 False
                checkpoint_path=local_weights_path  # 提供你下载的 .bin 文件路径
            )
            # using the weights from official vit-b, online
            # backbone_model = timm.create_model("vit_base_patch16_224", pretrained=True)

            #debug;test the backbone:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # dummy_input = torch.randn(2, 3, 224, 224).to(device)
            # with torch.no_grad():
            #     features = backbone_model(dummy_input)
            # print(f"模型输出了 {len(features)} 个特征图。")
            # for i, f in enumerate(features):
            #     print(f"  特征图 {i} 的形状: {f.shape}")
        elif backbone=="ViTB-384":
            backbone_model = timm.create_model('vit_base_patch16_rope_reg1_gap_256.sbb_in1k',pretrained=True,num_classes=0,imgsize2net=384)
            output_channel = 768
        elif backbone=="SwinB-224":
            backbone_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2S-256":
            backbone_model = timm.create_model("swinv2_small_window8_256", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2T-256":
            backbone_model = timm.create_model("swinv2_tiny_window16_256", pretrained=True)
            output_channel = 768
        elif backbone=="Convnext-T":
            backbone_model = timm.create_model("convnext_tiny", pretrained=True)
            output_channel = 768
        elif backbone=="EfficientNet-B2":
            backbone_model = timm.create_model("efficientnet_b2", pretrained=True)
            output_channel = 1408
        elif backbone=="EfficientNet-B3":
            backbone_model = timm.create_model("efficientnet_b3", pretrained=True)
            output_channel = 1536
        elif backbone=="EfficientNet-B5":
            backbone_model = timm.create_model("tf_efficientnet_b5", pretrained=True)
            output_channel = 2048
        elif backbone=="EfficientNet-B6":
            backbone_model = timm.create_model("tf_efficientnet_b6", pretrained=True)
            output_channel = 2304
        elif backbone=="vgg16":
            backbone_model = timm.create_model("vgg16", pretrained=True)
            output_channel = 512
        elif backbone=="cvt13":
            backbone_model, channels = get_cvt_models(model_size="cvt13")
            output_channel = channels[-1]
            checkpoint_weight = "pytorch-image-models/CvT-13-384x384-IN-22k.pth"
            backbone_model = self.load_checkpoints(checkpoint_weight, backbone_model)
        elif ('dino' in backbone) and ('v2' in backbone) : #added by zwk
            from .dinov2 import DINOv2,DINOV2_ARCHS
            backbone_model = DINOv2(backbone) #**self.opt.backbone_config) #backbone_config is a dict, todo:将网络相关参数包装为字典
            output_channel = DINOV2_ARCHS[backbone]
        elif ('dino'in backbone) and ("v3" in  backbone):
            output_channel = 1024
            from models.Backbone.dinov3.vision_transformer import vit_large
            backbone_model = vit_large(
                patch_size=16,
                # 启用特殊组件的关键参数
                layerscale_init=1.0e-05,
                n_storage_tokens=4,
                mask_k_bias=True,
                untie_global_and_local_cls_norm=True,
                # 其他在 backbones.py 中定义的默认值
                qkv_bias=True,
                drop_path_rate=0.0,
                norm_layer="layernormbf16",
                ffn_layer="mlp",
                ffn_bias=True,
                proj_bias=True
            )
            local_weights_path = "/home/data/zwk/ckpt_dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            checkpoint = torch.load(local_weights_path, map_location='cpu')
            backbone_model.load_state_dict(checkpoint, strict=False)

            # REPO_DIR = '/home/data/zwk/ckpt_dinov3/dinov3-main'
            # backbone_model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local',
            #                                 weights=local_weights_path )

            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # dummy_input = torch.randn(2, 3, 224, 224).to(device)
            # with torch.no_grad():
            #     backbone_model = backbone_model.to(device)
            #     features = backbone_model(dummy_input)
            # print(f"模型输出了 {len(features)} 个特征图。")
            # for i, f in enumerate(features):
            #     print(f"  特征图 {i} 的形状: {f.shape}")
        elif 'convnext' in backbone:
            # from transformers import ConvNextModel
            # backbone_model = ConvNextModel.from_pretrained("/home/data/zwk/convnext_base_22k_224.pth")
            from models.Backbone.convnext import convnext_base
            backbone_model = convnext_base(pretrained=True,in_22k=False)
            backbone_model.norm = nn.LayerNorm(512, eps=1e-6)  # final norm layer)
            backbone_model.last_stage = 3
            output_channel = 1024
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel
    
    def load_checkpoints(self, checkpoint_path, model):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        filter_ckpt = {k: v for k, v in ckpt.items() if "pos_embed" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filter_ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        return model

    def forward(self, image):
        features = self.backbone.forward_features(image)
        if ('dino' in self.backbone_name) and ("v3" in self.backbone_name): #for handling dinov3
            x_norm_clstoken = features["x_norm_clstoken"]
            x_norm_patchtokens = features["x_norm_patchtokens"]
            features = torch.concatenate([x_norm_clstoken.unsqueeze(1), x_norm_patchtokens], dim=1)
        return features