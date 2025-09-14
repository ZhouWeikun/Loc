# from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity
# from pytorch_metric_learning.utils.common_functions import pos_inf, neg_inf
from mertic_learning import LpDistance,DotProductSimilarity
from mertic_learning import pos_inf, neg_inf
import torch

"""
    #a example to use the ms-similiarity loss
        rc_dist = self.ms_loss_computer.l2_dist_computer(uav_rc,torch.concat([sat_rc_p,sat_rc_randoms],dim=0))
        rc_radius = self.trainer.train_dataloader.dataset.sat_rc_radius_normalized
        pos_mask = rc_dist < rc_radius
        feat_mat = self.ms_loss_computer.l2_dist_computer(feat[:N], feat[N:])
        ms_loss = self.ms_loss_computer.compute_loss(feat_mat,pos_mask,'min')
"""

class MSLossComputer(object):
    def __init__(self, alpha=2, beta=5, dtype=torch.float32,sim_metric='cos',**kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_ignore = torch.tensor(pos_inf(dtype),dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype),dtype=dtype)
        self.l2_dist_computer = LpDistance(normalize_embeddings=False, p=2)
        self.l1_dist_computer = LpDistance(normalize_embeddings=False, p=1)
        self.cossim_computer = DotProductSimilarity(normalize_embeddings=False)
        self.sim_metirc = sim_metric
        #todo: add the margin whening deciding hard neg and pos paris

    def compute_scl_loss(self,feat_mat,geo_weight_pos,geo_weight_neg,mining=True):
        #todo: add the func that computing loss according to self.sim_metirc
        if mining: #data mining:
            pos_mask = geo_weight_pos > 1e-9
            neg_mask = ~pos_mask

            pos_mat = feat_mat.clone()
            neg_mat = feat_mat.clone()
            pos_mat[neg_mask] = self.neg_ignore
            neg_mat[pos_mask] = self.pos_ignore

            pos_most_hard_per_row = torch.max(pos_mat,dim=-1)
            neg_most_hard_per_row = torch.min(neg_mat,dim=-1)
            hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask
            hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask

            # compute loss for hard_pos
            feat2compute_pos_loss = feat_mat * geo_weight_pos
            feat2compute_pos_loss = feat2compute_pos_loss.masked_fill(~hard_pos_mask,self.neg_ignore)
            feat2compute_pos_loss = torch.cat([feat2compute_pos_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            pos_loss = 1/self.alpha * torch.logsumexp(self.alpha * feat2compute_pos_loss, dim=-1, keepdim=True).mean()

            # compute loss for hard_neg
            feat2compute_neg_loss = feat_mat * geo_weight_neg
            feat2compute_neg_loss = feat2compute_neg_loss.masked_fill(~hard_neg_mask,self.pos_ignore)
            feat2compute_neg_loss = torch.cat([feat2compute_neg_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            neg_loss = 1/self.beta * torch.logsumexp(-self.beta * feat2compute_neg_loss, dim=-1, keepdim=True).mean()
            return pos_loss+neg_loss

            # debug,visualize the grad of the featmat:
            # feat_mat.retain_grad()
            # neg_loss.backward()
            # feat_mat_grad_neg_np = feat_mat.grad.detach().cpu().numpy()
            # # pos_loss.backward()
            # # feat_mat_grad_pos_np = feat_mat.grad.detach().cpu().numpy()
            # from matplotlib import  pyplot as plt
            # fig,ax = plt.subplots()
            # cax = ax.imshow(feat_mat_grad_neg_np[:,:16*8])
            # fig.colorbar(cax, ax=ax)  # 添加颜色条
            # plt.show()

        else: #without data mining:
            feat2compute_pos_loss = feat_mat * geo_weight_pos
            feat2compute_pos_loss = feat2compute_pos_loss.masked_fill(geo_weight_pos<1e-6, self.neg_ignore)
            feat2compute_pos_loss = torch.cat(
            [feat2compute_pos_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            pos_loss = 1 / self.alpha * torch.logsumexp(self.alpha * feat2compute_pos_loss , dim=-1, keepdim=True).mean()
            feat2compute_neg_loss = feat_mat * geo_weight_neg
            feat2compute_neg_loss = feat2compute_neg_loss.masked_fill(geo_weight_neg<1e-6, self.pos_ignore)
            feat2compute_neg_loss = torch.cat(
                [feat2compute_neg_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            neg_loss = 1 / self.beta * torch.logsumexp(-self.beta * feat2compute_neg_loss, dim=-1, keepdim=True).mean()
            return  pos_loss+neg_loss


    def compute_ms_loss(self,feat_mat, pos_mask, return_hard_mask=False):
        """
        Args:
            feat_mat ():
            pos_mask ():
            feat_metric ():  when the metric = similiarity, the value is 'max', when the metric = distance, the value is 'min'
            return_hard_mask ():
        Returns:

        """
        # hard pair degging:
        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()
        
        if self.sim_metirc == 'cos':
            pos_mat[neg_mask] = self.pos_ignore
            neg_mat[pos_mask] = self.neg_ignore
            
            pos_most_min_per_row = torch.min(pos_mat,dim=-1)
            neg_most_max_per_row = torch.max(neg_mat,dim=-1)
            hard_pos_mask = torch.lt(feat_mat, neg_most_max_per_row[0].unsqueeze(1))*pos_mask
            hard_neg_mask = torch.gt(feat_mat, pos_most_min_per_row[0].unsqueeze(1))*neg_mask

            # compute loss for hard_pos
            feat2compute_pos_loss = feat_mat.masked_fill(~hard_pos_mask, self.pos_ignore)
            feat2compute_pos_loss = torch.cat(
                [feat2compute_pos_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            pos_loss = 1 / self.alpha * torch.logsumexp(-self.alpha * feat2compute_pos_loss, dim=-1,
                                                        keepdim=True).mean()

            feat2compute_neg_loss = feat_mat.masked_fill(~hard_neg_mask, self.neg_ignore)
            feat2compute_neg_loss = torch.cat(
                [feat2compute_neg_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            neg_loss = 1 / self.beta * torch.logsumexp(self.beta * feat2compute_neg_loss, dim=-1, keepdim=True).mean()
        else:
            pos_mat[neg_mask] = self.neg_ignore
            neg_mat[pos_mask] = self.pos_ignore

            pos_most_hard_per_row = torch.max(pos_mat,dim=-1)   #find the maximum distance of positive value in every row
            neg_most_hard_per_row = torch.min(neg_mat,dim=-1)   #find the minimum distance of negative value in every row
            hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask #get the mask for positives which's distance > negatives
            hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask

            # compute loss for hard_pos
            feat2compute_pos_loss = feat_mat.masked_fill(~hard_pos_mask,self.neg_ignore)
            feat2compute_pos_loss = torch.cat([feat2compute_pos_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            pos_loss = 1/self.alpha * torch.logsumexp(self.alpha * feat2compute_pos_loss, dim=-1, keepdim=True).mean()
            # pos_loss = torch.logsumexp(self.alpha * feat2compute_pos_loss, dim=-1, keepdim=True).mean()

            feat2compute_neg_loss = feat_mat.masked_fill(~hard_neg_mask,self.pos_ignore)
            feat2compute_neg_loss = torch.cat([feat2compute_neg_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            neg_loss = 1/self.beta * torch.logsumexp(-self.beta * feat2compute_neg_loss, dim=-1, keepdim=True).mean()
            # neg_loss =  torch.logsumexp(-self.beta * feat2compute_neg_loss, dim=-1, keepdim=True).mean()

            # feat2compute_pos_loss_np = feat2compute_pos_loss.detach().cpu().numpy()
            # feat2compute_neg_loss_np = feat2compute_neg_loss.detach().cpu().numpy()
            # feat_mat_np = feat_mat.detach().cpu().numpy()

        loss = pos_loss+neg_loss
        if return_hard_mask:
            return loss,torch.logical_or(hard_neg_mask,hard_pos_mask)
        return loss