from nis import match

import torch
from torch import nn
# from torch.distributions.constraints import positive

from .TripletLoss import SameDomainTripletLoss, WeightedSoftTripletLoss, HardMiningTripletLoss, TripletLoss
from .FocalLoss import FocalLoss
# from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
import torch.nn.functional as F
from torch.autograd import Variable
from .MSLoss import MSLossComputer
from .InfoNceLoss import SampleInfoNCE

class Loss(nn.Module):
    def __init__(self, opt) -> None:
        super(Loss,self).__init__()
        self.opt = opt

        # 对比损失
        self.feature_loss_dict = {}
        for feat_loss in opt.feature_loss:
            if feat_loss == "TripletLoss":
                self.feature_loss_dict["TripletLoss"] = TripletLoss(margin=0.3, normalize_feature=True)
            elif feat_loss == "HardMiningTripletLoss":
                self.feature_loss_dict["HardMiningTripletLoss"]=HardMiningTripletLoss(margin=0.3, normalize_feature=True)
            elif feat_loss == "SameDomainTripletLoss":
                self.feature_loss_dict["SameDomainTripletLoss"]=SameDomainTripletLoss(margin=0.3)
            elif feat_loss == "WeightedSoftTripletLoss":
                self.feature_loss_dict["WeightedSoftTripletLoss"] = WeightedSoftTripletLoss()
            elif feat_loss == "ContrastiveLoss":
                self.feature_loss_dict["ContrastiveLoss"] = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
            elif feat_loss == "MSLoss":
                self.feature_loss_dict["MSLoss"] = MSLossComputer(alpha=1,beta=1,dtype=torch.float16 if opt.autocast else torch.float32,sim_metric='cos')#alpha=0.02,beta=0.04,
            elif feat_loss == "InfoNceLoss":
                self.feature_loss_dict["InfoNceLoss"] = SampleInfoNCE(temperature=0.1,sim_mertic='cos',neg_from_pos=True)

        # if opt.feature_loss == "TripletLoss":
        #     self.feature_loss = TripletLoss(margin=0.3, normalize_feature=True)
        # elif opt.feature_loss == "HardMiningTripletLoss":
        #     self.feature_loss = HardMiningTripletLoss(margin=0.3, normalize_feature=True)
        # elif opt.feature_loss == "SameDomainTripletLoss":
        #     self.feature_loss = SameDomainTripletLoss(margin=0.3)
        # elif opt.feature_loss == "WeightedSoftTripletLoss":
        #     self.feature_loss = WeightedSoftTripletLoss()
        # elif opt.feature_loss == "ContrastiveLoss":
        #     self.feature_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        # else:
        #     self.feature_loss = None

        if opt.w_classify == True:
            # 分类损失
            if opt.cls_loss == "CELoss":
                self.cls_loss = nn.CrossEntropyLoss()
            elif opt.cls_loss == "FocalLoss":
                self.cls_loss = FocalLoss(alpha=0.25, gamma=2, num_classes = opt.nclasses)
            else:
                self.cls_loss = None

            # KL 损失
            if opt.kl_loss == "KLLoss":
                self.kl_loss = nn.KLDivLoss(reduction='batchmean')
            else:
                self.kl_loss = None

    def forward(self, outputs, outputs2, labels, labels2): #outputs->uav,outputs2->sat
        if self.opt.w_classify:
            cls1,feature1 = outputs
            cls2,feature2 = outputs2
            loss = 0

            # 特征对比损失
            feat_loss = torch.tensor((0))
            if self.feature_loss is not None:
                # split_num = self.opt.batchsize // self.opt.sample_num
                feat_loss = self.calc_triplet_loss(
                    feature1, feature2, labels, self.feature_loss)
                loss += feat_loss

            # 分类损失
            res_cls_loss = torch.tensor((0))
            if self.cls_loss is not None:
                res_cls_loss = self.calc_cls_loss(cls1, labels, self.cls_loss) + \
                    self.calc_cls_loss(cls2, labels2, self.cls_loss)
                loss += res_cls_loss

            # 相互学习
            res_kl_loss = torch.tensor((0))
            if self.kl_loss is not None:
                res_kl_loss = self.calc_kl_loss(cls1, cls2, self.kl_loss)
                loss += res_kl_loss

            # if self.opt.epoch < self.opt.warm_epoch:
            #     warm_up = 0.1  # We start from the 0.1*lrRate
            #     warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
            #     warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            #     loss *= warm_up

            return loss, res_cls_loss, feat_loss, res_kl_loss
        else:
            # 特征对比损失
            feat_loss2return = {}
            for key,loss_func in self.feature_loss_dict.items():
                if key in ["TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"]:
                    feat_loss2return['TripletLoss'] = self.calc_triplet_loss( outputs, outputs2, labels, loss_func)
                if key in ['MSLoss']:
                    feat1 = torch.cat(outputs, dim=1)
                    feat1 = feat1/torch.norm(feat1,dim=1,keepdim=True)
                    feat2 = torch.cat(outputs2, dim=1)
                    feat2 = feat2 / torch.norm(feat2, dim=1, keepdim=True)
                    feat_mat = self.feature_loss_dict['MSLoss'].l2_dist_computer(feat1, feat2)
                    positive_mask = torch.eye(feat1.shape[0], dtype=torch.int, device=feat_mat.device).bool()
                    feat_loss2return['MSLoss'] = self.feature_loss_dict['MSLoss'].compute_ms_loss(feat_mat, positive_mask)
                if key in ['InfoNceLoss']:
                    if isinstance(outputs, list):
                        loss = 0
                        for i in range(len(outputs)):
                            # out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
                            # labels_concat = torch.cat((labels, labels), dim=0)
                            # loss += loss_func(out_concat, labels_concat)
                            loss += self.feature_loss_dict['InfoNceLoss'].forward(outputs[i], outputs2[i])
                        feat_loss2return['InfoNceLoss'] = loss / len(outputs)
                    # else:
                    # feat1 = torch.cat(outputs, dim=1)
                    # feat2 = torch.cat(outputs2, dim=1)
                    # feat_loss2return['InfoNceLoss'] = self.feature_loss_dict['InfoNceLoss'].forward(feat1,feat2)

            return feat_loss2return

    def calc_cls_loss(self, outputs, labels, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in outputs:
                loss += loss_func(i, labels)
            loss = loss/len(outputs)
        else:
            loss = loss_func(outputs, labels)
        return loss

    def calc_kl_loss(self, outputs, outputs2, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in range(len(outputs)):
                loss += loss_func(F.log_softmax(outputs[i], dim=1),
                                F.softmax(Variable(outputs2[i]), dim=1))
            loss = loss/len(outputs)
        else:
            loss = loss_func(F.log_softmax(outputs, dim=1),
                            F.softmax(Variable(outputs2), dim=1))
        return loss


    def calc_triplet_loss(self, outputs, outputs2, labels, loss_func, split_num=8):
        if isinstance(outputs, list):
            loss = 0
            for i in range(len(outputs)):
                out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
                labels_concat = torch.cat((labels, labels), dim=0)
                loss += loss_func(out_concat, labels_concat)
            loss = loss/len(outputs)
        else:
            out_concat = torch.cat((outputs, outputs2), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss = loss_func(out_concat, labels_concat)
        return loss


