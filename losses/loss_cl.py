from nis import match

import torch
from torch import nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.distributions.constraints import positive
from mertic_learning import LpDistance,DotProductSimilarity #会导致numpy版本改变
# from .TripletLoss import SameDomainTripletLoss, WeightedSoftTripletLoss, HardMiningTripletLoss, TripletLoss
# from .FocalLoss import FocalLoss
# from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
from .MSLoss import MSLossComputer
from .InfoNceLoss import SampleInfoNCE
from .WeightedSoftTripletLoss import WeightedSoftTripletLoss_v3,WeightedSoftTripletLoss_v2,WeightedSoftTripletLoss_v1,WeightedSoftTripletLoss_v0


class Loss(nn.Module):
    def __init__(self, opt) -> None:
        super(Loss,self).__init__()
        self.opt = opt
        self.cossim_computer = DotProductSimilarity(normalize_embeddings=True)
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.eucdist_computer_rc = LpDistance(normalize_embeddings=False)

        # 对比损失
        self.feature_loss_dict = {}
        for feat_loss in opt.feature_loss:
            if feat_loss == "WeightedSoftTripletLoss_v3":
                self.feature_loss_dict["WeightedSoftTripletLoss_v3"] = WeightedSoftTripletLoss_v3(alpha=3, dtype=opt.loss_dtype)
            elif feat_loss == "WeightedSoftTripletLoss_v2":
                self.feature_loss_dict["WeightedSoftTripletLoss_v2"] = WeightedSoftTripletLoss_v2(alpha=3, dtype=opt.loss_dtype)
            elif feat_loss == "WeightedSoftTripletLoss_v1":
                self.feature_loss_dict["WeightedSoftTripletLoss_v1"] = WeightedSoftTripletLoss_v1(alpha=3, dtype=opt.loss_dtype)
            elif feat_loss == "WeightedSoftTripletLoss_v0":
                self.feature_loss_dict["WeightedSoftTripletLoss_v0"] = WeightedSoftTripletLoss_v0(alpha=3, dtype=opt.loss_dtype)
            elif feat_loss == "MSLoss":
                self.feature_loss_dict["MSLoss"] = MSLossComputer(alpha=1, beta=1, sim_metric='l2', dtype=opt.loss_dtype)#alpha=0.02,beta=0.04,
            elif feat_loss == "InfoNceLoss":
                self.feature_loss_dict["InfoNceLoss"] = SampleInfoNCE(temperature=0.1,sim_mertic='cos',neg_from_pos=True)


    def forward(self, feats_q, feats_p, feats_rand, rcs_query=None,rcs_pos=None, rcs_rand=None,rc_radius=None):
        loss2return = {}
        loss_all = 0

        for key in self.feature_loss_dict.keys():
            if "WeightedSoftTripletLoss" in key:
                wtl_loss = self.feature_loss_dict[key].forward(feats_q, feats_p, feats_rand,
                                                                                     rcs_query, rcs_pos, rcs_rand,
                                                                                     rc_radius
                                                                                     )
                loss2return[key] = wtl_loss
                loss_all += wtl_loss
            # if key == "WeightedSoftTripletLoss_v3":
            #     wtl_loss = self.feature_loss_dict["WeightedSoftTripletLoss_v3"].forward(feats_q, feats_p, feats_rand,
            #                                                                          rcs_query, rcs_pos, rcs_rand,
            #                                                                          rc_radius
            #                                                                          )
            #
            #     loss2return["WeightedSoftTripletLoss_v3"] = wtl_loss
            #     loss_all += wtl_loss
            # if key == "WeightedSoftTripletLoss_v2":
            #     wtl_loss = self.feature_loss_dict["WeightedSoftTripletLoss_v2"].forward(feats_q, feats_p, feats_rand,
            #                                                                          rcs_query, rcs_pos, rcs_rand,
            #                                                                          rc_radius
            #                                                                          )
            #
            #     loss2return["WeightedSoftTripletLoss_v2"] = wtl_loss
            #     loss_all += wtl_loss
            # if key == "WeightedSoftTripletLoss_v0":
            #     wtl_loss = self.feature_loss_dict["WeightedSoftTripletLoss_v0"].forward(
            #                                                                          feats_q=feats_q,
            #                                                                          feats_p=feats_p,
            #                                                                          feats_rand=feats_rand,
            #                                                                          )
            #     loss2return["WeightedSoftTripletLoss_v0"] = wtl_loss
            #     loss_all += wtl_loss
            if key == "MSLoss":
                # version 0:
                rc_dist_mat = self.eucdist_computer_rc(rcs_query, torch.concat([rcs_pos, rcs_rand], dim=0))
                pos_mask = rc_dist_mat < rc_radius
                feat_mat = self.eucdist_computer_feat(feats_q, torch.concat([feats_p, feats_rand], dim=0))
                # version 1:
                # rc_dist_mat = self.eucdist_computer(torch.cat([rcs_query,rcs_pos],dim=0), torch.concat([rcs_query, rcs_pos, rcs_rand], dim=0))
                # pos_mask = rc_dist_mat < rc_radius
                # feat_mat = self.eucdist_computer(torch.cat([feats_q,feats_p]), torch.concat([feats_q, feats_p, feats_rand], dim=0))
                ms_loss = self.feature_loss_dict["MSLoss"].compute_ms_loss(feat_mat, pos_mask.to(feats_q.device))
                loss2return["MSLoss"] = ms_loss
                loss_all += ms_loss

                #debug:
                # rc_dist_mat_np = rc_dist_mat.detach().cpu().numpy()
                # feat_mat_np = feat_mat.detach().cpu().numpy()

        loss2return['all'] = loss_all
        return loss2return


    def forward_org(self, outputs, outputs2, labels): #outputs->uav; outputs2->sat
        # 特征对比损失
        feat_loss2return = {}
        for key,loss_func in self.feature_loss_dict.items():
            if key in ["TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"]:
                # feat_loss2return['TripletLoss'] = self.calc_triplet_loss( outputs, outputs2, labels, loss_func)
                out_concat = torch.cat((outputs, outputs2), dim=0)
                labels_concat = torch.cat((labels, labels), dim=0)
                loss = loss_func(out_concat, labels_concat)

                feat_loss2return['TripletLoss'] = loss
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
                else:
                    feat1 = torch.cat(outputs, dim=1)
                    feat2 = torch.cat(outputs2, dim=1)
                    feat_loss2return['InfoNceLoss'] = self.feature_loss_dict['InfoNceLoss'].forward(feat1,feat2)

        return feat_loss2return


    # def calc_triplet_loss(self, outputs, outputs2, labels, loss_func, split_num=8):
    #     if isinstance(outputs, list):
    #         loss = 0
    #         for i in range(len(outputs)):
    #             out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
    #             labels_concat = torch.cat((labels, labels), dim=0)
    #             loss += loss_func(out_concat, labels_concat)
    #         loss = loss/len(outputs)
    #     else:
    #         out_concat = torch.cat((outputs, outputs2), dim=0)
    #         labels_concat = torch.cat((labels, labels), dim=0)
    #         loss = loss_func(out_concat, labels_concat)
    #     return loss


