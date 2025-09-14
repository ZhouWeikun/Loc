import timm
import torch

class GTAVIT(object):
    def __init__(self,p2chpt= '/home/data/zwk/ckpt_game4loc/vit_base_eva_gta_cross_area.pth',device=None):
        self.model = get_gta_vit(p2chpt)
        # if torch.cuda.is_available():
        #     device = torch.device("cuda:0")
        # else:
        #     device = torch.device("cpu")
        self.device = device
        self.model.to(device) if device is not None else None
        for param in self.model.parameters():
            param.requires_grad = False



def get_gta_vit(p2checkpoint):
    img_encoder = timm.create_model('vit_base_patch16_rope_reg1_gap_256.sbb_in1k', pretrained=False, num_classes=0,
                                         img_size=384)
    checkpoint = torch.load(p2checkpoint)
    prefix = 'model.'
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith(prefix):
            # 移除前缀 'backbone.model.'
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v
    img_encoder.load_state_dict(new_state_dict, strict=False)
    return img_encoder
