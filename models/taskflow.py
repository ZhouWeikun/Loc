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
        if (type(opt.head)==str) and (len(opt.head)>0):
            self.head = make_head(opt)

    def forward(self, image,ret_patch_token=False,ret_cls_token=False):
        features = self.backbone(image)
        res = self.head(features) if hasattr(self, 'head') else features

        if (not ret_patch_token) and (not ret_cls_token):
            return res
        elif ret_cls_token and (not ret_patch_token):
            return (res,features[:, 0])
        elif not ret_cls_token and ret_patch_token:
            return (res,features[:, 1:])
        else:
            return (res, features[:, 0], features[:, 1:])


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
