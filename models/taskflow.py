import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import torch
import math
# from train_img_encoder.util_circorr_fm_radon import RadonHandler


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = make_backbone(opt)
        opt.in_planes = self.backbone.output_channel
        self.opt = opt
        if opt.head != "":
            self.head = make_head(opt)

    def forward(self, image,ret_sg=False,ret_cls_token=False):
        features = self.backbone(image)
        res = self.head(features) if self.opt.head != "" else None
        cls_token = features[:, 0]

        #ready to return sg:
        if ret_sg:
            if not hasattr(self, 'radon'):
                self.radon = RadonHandler(device=features.device)
            with torch.no_grad():
                featmaps = features[:, 1:].permute(0, 2, 1).contiguous()
                featmaps = featmaps.reshape(featmaps.shape[0],-1,int(math.sqrt(featmaps.shape[2])),int(math.sqrt(featmaps.shape[2])))
                featmaps_mean = featmaps.mean(dim=1, keepdim=True)
                sgs = self.radon.forward(featmaps_mean).squeeze()
                return res,sgs
        if self.opt.head == "":
            return cls_token
        else:
            if ret_cls_token:
                return res,cls_token
            else:
                return res

    def load_params(self, load_from):
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)


def make_img_encoder(opt):
    model = Model(opt)
    # if os.path.exists(opt.load_from):
    #     model.load_params(opt.load_from)
    return model
