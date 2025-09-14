import torch.nn as nn
from .SingleBranch import SingleBranch, SingleBranchCNN, SingleBranchSwin
from .FSRA import FSRA, FSRA_CNN, FSRA_wo_CLS
from .LPN import LPN, LPN_CNN
from .GeM import GeM
from .NetVLAD import NetVLAD
from .salad import SALAD

def make_head(opt):
    return Head(opt)


class Head(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.head = self.init_head(opt)
        self.opt = opt

    def init_head(self, opt):
        head = opt.head
        if head == "SingleBranch":
            head_model = SingleBranch(opt)
        elif head == "SingleBranchCNN":
            head_model = SingleBranchCNN(opt)
        elif head == "SingleBranchSwin":
            head_model = SingleBranchSwin(opt)
        elif head == "NetVLAD":
            head_model = NetVLAD(opt)
        elif head == "FSRA":
            head_model = FSRA(opt) if opt.w_classify else FSRA_wo_CLS(opt)
        elif head == "FSRA_CNN":
            head_model = FSRA_CNN(opt)
        elif head == "LPN":
            head_model = LPN(opt)
        elif head == "LPN_CNN":
            head_model = LPN_CNN(opt)
        elif head == "GeM":
            head_model = GeM(opt)
        elif "salad" in head.lower():
            head_model = SALAD()
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, features):
        features = self.head(features)
        return features
