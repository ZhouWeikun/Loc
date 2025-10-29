import torch
from torch import nn
# from pytorch_metric_learning.distances import LpDistance, DotProductSimilarity
from mertic_learning import LpDistance,DotProductSimilarity
# from mertic_learning import pos_inf, neg_inf
#todo:decoupling from mertic_learning

def pos_inf(dtype):
    return torch.finfo(dtype).max

def neg_inf(dtype):
    return torch.finfo(dtype).min

"""
version 0: 
"""
class WeightedSoftTripletLoss_v0(nn.Module):
    """
    only support one positive pair
    """
    # writed by zwk
    def __init__(self,alpha=3,**kwargs):
        super(WeightedSoftTripletLoss_v0, self).__init__()
        self.alpha = alpha
        self.eucdist_computer = LpDistance(normalize_embeddings=True)

    def forward(self, feats_q, feats_p, feats_rand=None,
                rcs_query=None, rcs_pos=None,
                rcs_rand=None, rc_radius=None
                ):
        """
        Args:
            feats_q:feats from uavimgs, [B,C,H,W]
            feats_p:feats from satimgs_positive, [B,C,H,W]
            feats_rand:feats from satimgs_random, [B,C,H,W]
        """
        ids_d = torch.arange(feats_q.shape[0])
        ids = torch.concatenate([ids_d, ids_d])
        feats_r = torch.concatenate([feats_q,feats_p])
        feats_c = torch.concatenate([feats_q,feats_p])

        fdist_mat = self.eucdist_computer(feats_r,feats_c)
        # fdist_mat_np = fdist_mat.detach().cpu().numpy()

        # hard minning
        N = len(ids)
        is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_r.shape[0]]
        dist_hardp, relative_p_inds = torch.max(fdist_mat[is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        dist_hardn, relative_n_inds = torch.min(fdist_mat[~is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn))).mean() #(dist_hardp - dist_hardn) large -> loss large

        """
        acturally, loss = torch.log(1 + torch.exp(self.alpha  * -(dist_hardn - dist_hardp))) = -log(sigmoid(-(alpha  * (dist_hardn - dist_hardp))))
        dist_hardn - dist_hardp large -> loss small
        loss = torch.log(1 + torch.exp(-self.alpha * (dist_hardn-dist_hardp))).mean()
        torch.stack([dist_hardn-dist_hardp,torch.log(1 + torch.exp(-self.alpha * (dist_hardn-dist_hardp)))],dim=-1)
        torch.sigmoid( dist_hardn-dist_hardp)
        """

        return loss


"""
version 1: add rand_samples as negative references based on version 0
"""
class WeightedSoftTripletLoss_v1(nn.Module):
    # writed by zwk
    def __init__(self,alpha=3,**kwargs):
        super(WeightedSoftTripletLoss_v1, self).__init__()
        self.alpha = alpha
        self.eucdist_computer = LpDistance(normalize_embeddings=True)

    def forward(self, feats_q, feats_p, feats_rand,
                rcs_query=None, rcs_pos=None,
                rcs_rand=None, rc_radius=None
                ):
        """
        Args:
            feats_q:feats from uavimgs, [B,C,H,W]
            feats_p:feats from satimgs_positive, [B,C,H,W]
            feats_rand:feats from satimgs_random, [B,C,H,W]
        """
        ids_d = torch.arange(feats_q.shape[0])
        ids_s_rand = torch.arange(feats_rand.shape[0]) + ids_d.shape[0] * 2
        ids = torch.concatenate([ids_d, ids_d, ids_s_rand])
        feats_r = torch.concatenate([feats_q,feats_p])
        feats_c = torch.concatenate([feats_q,feats_p,feats_rand])
        fdist_mat = self.eucdist_computer(feats_r,feats_c)

        # hard minning
        N = len(ids)
        is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_r.shape[0]]
        dist_hardp, relative_p_inds = torch.max(fdist_mat[is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        dist_hardn, relative_n_inds = torch.min(fdist_mat[~is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn))).mean() #(dist_hardp - dist_hardn) large -> loss large

        return loss


class MSLoss_fm_mat(nn.Module):
    # writed by zwk
    def __init__(self,dtype=torch.float32,slope_pos=7.5, slope_neg=7.5,tau=0.5, **kwargs):
        super(MSLoss_fm_mat, self).__init__()
        self.pos_ignore = torch.tensor(pos_inf(dtype),dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype),dtype=dtype)
        self.alpha = slope_pos
        self.beta = slope_neg
        self.tau = tau

    def forward(self, feat_mat,
                pos_mask,
                metric='dist',
                ):
        """
        Args:
            feats_q:feats from uavimgs, [B,C,H,W]
            feats_p:feats from satimgs_positive, [B,C,H,W]
            feats_rand:feats from satimgs_random, [B,C,H,W]
        """

        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()

        if metric=='dist':
            # hard minning:
            pos_mat[neg_mask] = self.neg_ignore #fill the neg with -inf
            neg_mat[pos_mask] = self.pos_ignore #fill the pos with +inf

            pos_most_hard_per_row = torch.max(pos_mat,dim=-1)   #find the maximum distance of positive value in every row
            neg_most_hard_per_row = torch.min(neg_mat,dim=-1)   #find the minimum distance of negative value in every row
            hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask #getting the hard_pos_mask by comparing the neg_most_hard_per_row to the pos
            hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask #getting the hard_neg_mask by comparing the pos_most_hard_per_row to the neg

            # compute MS-loss:
                # loss for hard_pos
            hard_pos2compute_loss = feat_mat.masked_fill(~hard_pos_mask,self.neg_ignore) #fill the easy_pos_samples with -inf
            hard_pos2compute_loss = self.alpha * (hard_pos2compute_loss - self.tau)
            hard_pos2compute_loss = torch.cat([hard_pos2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_pos = 1 / self.alpha * torch.logsumexp(hard_pos2compute_loss, dim=-1, keepdim=True).mean()
            # loss_pos = 1/self.alpha * torch.logsumexp(self.alpha * (hard_pos2compute_loss - self.tau), dim=-1, keepdim=True).mean()
                # loss for hard_neg
            hard_neg2compute_loss = feat_mat.masked_fill(~hard_neg_mask,self.pos_ignore)
            hard_neg2compute_loss = -self.beta * (hard_neg2compute_loss - self.tau)
            hard_neg2compute_loss = torch.cat([hard_neg2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_neg = 1 / self.beta * torch.logsumexp(hard_neg2compute_loss, dim=-1, keepdim=True).mean()
            # loss_neg = 1/self.beta * torch.logsumexp(-self.beta * (hard_neg2compute_loss - self.tau), dim=-1, keepdim=True).mean()
        else:
            pos_mat[neg_mask] = self.pos_ignore
            neg_mat[pos_mask] = self.neg_ignore

            pos_most_min_per_row = torch.min(pos_mat, dim=-1)
            neg_most_max_per_row = torch.max(neg_mat, dim=-1)
            hard_pos_mask = torch.lt(feat_mat, neg_most_max_per_row[0].unsqueeze(1)) * pos_mask
            hard_neg_mask = torch.gt(feat_mat, pos_most_min_per_row[0].unsqueeze(1)) * neg_mask

            hard_pos2compute_loss = feat_mat.masked_fill(~hard_pos_mask,self.neg_ignore)
            hard_pos2compute_loss = self.alpha * (self.tau-hard_pos2compute_loss)
            hard_pos2compute_loss = torch.cat([hard_pos2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_pos = torch.logsumexp(hard_pos2compute_loss, dim=-1, keepdim=True).mean()

            hard_neg2compute_loss = feat_mat.masked_fill(~hard_neg_mask,self.pos_ignore)
            hard_neg2compute_loss = -self.beta * (hard_neg2compute_loss-self.tau)
            hard_neg2compute_loss = torch.cat([hard_neg2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_neg = torch.logsumexp(hard_neg2compute_loss, dim=-1, keepdim=True).mean()

        return loss_pos+loss_neg


class SWTLoss_fm_mat(nn.Module):
    # writed by zwk
    def __init__(self, dtype=torch.float32, slope_pos=5,slope_neg=5,decoupling=False, **kwargs):
        super(SWTLoss_fm_mat, self).__init__()
        self.pos_ignore = torch.tensor(pos_inf(dtype), dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype), dtype=dtype)
        self.alpha = slope_pos
        self.beta = slope_neg
        self.decoupling = decoupling
        self.weight_dist_func = lambda x:1/(1+torch.exp(-8.5*(x-0.15)))


    def forward(self, feat_mat,
                pos_mask,
                dist_mat=None,
                metric='dist',
                w_weight = False,
                ):

        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()

        if metric == 'dist':
            # hard minning:
            pos_mat[neg_mask] = self.neg_ignore  # fill the neg with -inf
            neg_mat[pos_mask] = self.pos_ignore  # fill the pos with +inf

            pos_most_hard_per_row = torch.max(pos_mat,dim=-1)  # find the maximum distance of positive value in every row
            neg_most_hard_per_row = torch.min(neg_mat,dim=-1)  # find the minimum distance of negative value in every row

            if not self.decoupling:
                if not w_weight:
                    return torch.log(1 + torch.exp( self.alpha * (pos_most_hard_per_row[0] - neg_most_hard_per_row[0]))).mean()  # (dist_hardp - dist_hardn) large -> loss large

                indices_hard_pos = pos_most_hard_per_row[1].unsqueeze(1)
                dist_pos_most_hard_per_row = torch.gather(dist_mat, dim=1, index=indices_hard_pos)
                indices_hard_neg = neg_most_hard_per_row[1].unsqueeze(1)
                dist_neg_most_hard_per_row = torch.gather(dist_mat, dim=1, index=indices_hard_neg)
                neg_weight = self.weight_dist_func(dist_neg_most_hard_per_row-dist_pos_most_hard_per_row).squeeze()
                return (neg_weight*torch.log(1 + torch.exp(self.alpha * (pos_most_hard_per_row[0] - neg_most_hard_per_row[0])))).mean()

            hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask #getting the hard_pos_mask by comparing the neg_most_hard_per_row to the pos
            hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask #getting the hard_neg_mask by comparing the pos_most_hard_per_row to the neg

                # loss for hard_pos
            delta_pos2compute_loss = feat_mat - neg_most_hard_per_row[0].unsqueeze(1)
            hard_pos2compute_loss = delta_pos2compute_loss.masked_fill(~hard_pos_mask, self.neg_ignore)
            hard_pos2compute_loss = self.alpha * hard_pos2compute_loss
            hard_pos2compute_loss = torch.cat([hard_pos2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_per_row_pos = torch.logsumexp(hard_pos2compute_loss, dim=-1)
            non_zero_mask_pos = loss_per_row_pos > 0
                # If there are any rows with loss, compute their mean
            if non_zero_mask_pos.sum() > 0:
                loss_pos = loss_per_row_pos[non_zero_mask_pos].mean()
            else:
                # Handle the case where all samples in the batch are easy
                loss_pos = torch.tensor(0.0, device=feat_mat.device)

                # loss for hard_neg
            delta_neg2compute_loss = pos_most_hard_per_row[0].unsqueeze(1) - feat_mat
            hard_neg2compute_loss = delta_neg2compute_loss.masked_fill(~hard_neg_mask,self.pos_ignore)
            hard_neg2compute_loss = -self.beta * hard_neg2compute_loss
            hard_neg2compute_loss = torch.cat([hard_neg2compute_loss, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
            loss_per_row_neg = torch.logsumexp(hard_neg2compute_loss, dim=-1)
            non_zero_mask_neg = loss_per_row_neg > 0
                # If there are any rows with loss, compute their mean
            if non_zero_mask_neg.sum() > 0:
                loss_neg = loss_per_row_neg[non_zero_mask_neg].mean()
            else:
                # Handle the case where all samples in the batch are easy
                loss_neg = torch.tensor(0.0, device=feat_mat.device)
        else:
            raise NotImplementedError("This function has not been implemented yet, baby is looking forward to it!")
        return loss_neg + loss_pos


"""
version 2: add neg_weight based on version 1
"""
class WeightedSoftTripletLoss_v2(nn.Module):
    # writed by zwk
    def __init__(self,alpha=3,**kwargs):
        super(WeightedSoftTripletLoss_v2, self).__init__()
        self.alpha = alpha
        self.eucdist_computer = LpDistance(normalize_embeddings=True)

    def forward(self, feats_q, feats_p, feats_rand,
                rcs_query=None, rcs_pos=None,
                rcs_rand=None, rc_radius=None
                ):
        """
        Args:
            feats_q:feats from uavimgs, [B,C,H,W]
            feats_p:feats from satimgs_positive, [B,C,H,W]
            feats_rand:feats from satimgs_random, [B,C,H,W]
        """
        ids_d = torch.arange(feats_q.shape[0])
        ids_s_rand = torch.arange(feats_rand.shape[0]) + ids_d.shape[0] * 2
        ids = torch.concatenate([ids_d, ids_d, ids_s_rand])
        feats_r = torch.concatenate([feats_q,feats_p])
        feats_c = torch.concatenate([feats_q,feats_p,feats_rand])
        fdist_mat = self.eucdist_computer(feats_r,feats_c)
        # fcos_mat = self.cossim_computer(feats_r,feats_c)

        # hard minning
        N = len(ids)
        is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_r.shape[0]]
        # is_pos_np = is_pos.detach().cpu().numpy()
        dist_hardp, relative_p_inds = torch.max( fdist_mat[is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        dist_hardn, relative_n_inds = torch.min( fdist_mat[~is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)

        # computing the weight of negivate pairs
        rcdist_mat = self.eucdist_computer(torch.concatenate([rcs_query,rcs_pos],dim=0),torch.concatenate([rcs_query,rcs_pos,rcs_rand],dim=0))
        # rcdist_mat_np = rcdist_mat.detach().cpu().numpy()
        rcdist_mat_rel = rcdist_mat/rc_radius
        # rcdist_mat_rel_np = rcdist_mat_rel.detach().cpu().numpy()
        neg_weight = torch.sigmoid(5* (rcdist_mat_rel-1.25)).to(fdist_mat.device)
        # neg_weight_np = neg_weight.detach().cpu().numpy()
        row_indices = torch.arange(fdist_mat.shape[0], dtype = torch.long, device=fdist_mat.device)
        hard_neg_weights = neg_weight[row_indices, relative_n_inds.squeeze(dim=1)]

        loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn) * hard_neg_weights )).mean() #(dist_hardp - dist_hardn) large -> loss large

        return loss





"""
version 3: including the random samples as query based on version 2
"""
class WeightedSoftTripletLoss_v3(nn.Module):
    # writed by zwk
    def __init__(self,alpha=3,dtype=torch.float32,**kwargs):
        super(WeightedSoftTripletLoss_v3, self).__init__()
        self.alpha = alpha
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=False)
        self.eucdist_computer_rc = LpDistance(normalize_embeddings=False)
        self.pos_ignore = torch.tensor(pos_inf(dtype),dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype),dtype=dtype)

    def forward(self, feats_q, feats_p, feats_rand,
                rcs_query=None, rcs_pos=None,
                rcs_rand=None, rc_radius=None
                ):
        """
        Args:
            feats_q:feats from uavimgs, [B,C,H,W]
            feats_p:feats from satimgs_positive, [B,C,H,W]
            feats_rand:feats from satimgs_random, [B,C,H,W]
        """
        ids_d = torch.arange(feats_q.shape[0])
        ids_s_rand = torch.arange(feats_rand.shape[0]) + ids_d.shape[0] * 2
        ids = torch.concatenate([ids_d, ids_d, ids_s_rand])
        # feats_r = torch.concatenate([feats_q,feats_p])
        feats_c = torch.concatenate([feats_q,feats_p,feats_rand])
        fdist_mat = self.eucdist_computer_feat(feats_c,feats_c)
        # fdist_mat_np =  fdist_mat.detach().cpu().numpy()

        # hard minning
        N = len(ids)
        is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_q.shape[0]*2]
        is_pos_rand = torch.eye(fdist_mat.shape[0],dtype=torch.int)[feats_q.shape[0]*2:].bool()
        is_pos = torch.concatenate([is_pos,is_pos_rand],dim=0)
        pos_mat = fdist_mat.clone()
        neg_mat = fdist_mat.clone()
        pos_mat[~is_pos] = self.neg_ignore #just keep the value of positive unchanged, others are -inf
        neg_mat[is_pos] = self.pos_ignore #just keep the value of positive unchanged, others are +inf
        dist_hardp,relative_p_inds = torch.max(pos_mat, dim=-1)  # find the maximum distance of positive value in every row
        dist_hardn,relative_n_inds = torch.min(neg_mat, dim=-1)  # find the minimum distance of negative value in every row

        # computing the weight of negivate pairs
        rcdist_mat = self.eucdist_computer_rc(torch.concatenate([rcs_query,rcs_pos,rcs_rand],dim=0),torch.concatenate([rcs_query,rcs_pos,rcs_rand],dim=0))
        # rcdist_mat_np = rcdist_mat.detach().cpu().numpy()
        rcdist_mat_rel = rcdist_mat/rc_radius
        # rcdist_mat_rel_np = rcdist_mat_rel.detach().cpu().numpy()
        neg_weight = torch.sigmoid(10* (rcdist_mat_rel-0.5)).to(fdist_mat.device) #version 1
        # neg_weight = torch.sigmoid(5* (rcdist_mat_rel-1.25)).to(fdist_mat.device) #version 0
        # neg_weight_np = neg_weight.detach().cpu().numpy()
        row_indices = torch.arange(fdist_mat.shape[0], dtype = torch.long, device=fdist_mat.device)
        hard_neg_weights = neg_weight[row_indices, relative_n_inds]

        loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn) * hard_neg_weights )) #(dist_hardp - dist_hardn) large -> loss large
        loss_p = loss[:feats_p.shape[0]*2].mean()
        loss_rand = loss[feats_p.shape[0]*2:].mean()

        return loss_p+loss_rand

# """
# version 4: including the random samples as query based on version 0
# """
# class WeightedSoftTripletLoss_v1(nn.Module):
#     # writed by zwk
#     def __init__(self,alpha=3,dtype=torch.float32,**kwargs):
#         super(WeightedSoftTripletLoss_v1, self).__init__()
#         self.alpha = alpha
#         self.eucdist_computer = LpDistance(normalize_embeddings=True)
#         self.pos_ignore = torch.tensor(pos_inf(dtype),dtype=dtype)
#         self.neg_ignore = torch.tensor(neg_inf(dtype),dtype=dtype)
#
#     def forward(self, feats_q, feats_p, feats_rand,
#                 rcs_query=None, rcs_pos=None,
#                 rcs_rand=None, rc_radius=None
#                 ):
#         """
#         Args:
#             feats_q:feats from uavimgs, [B,C,H,W]
#             feats_p:feats from satimgs_positive, [B,C,H,W]
#             feats_rand:feats from satimgs_random, [B,C,H,W]
#         """
#         ids_d = torch.arange(feats_q.shape[0])
#         ids_s_rand = torch.arange(feats_rand.shape[0]) + ids_d.shape[0] * 2
#         ids = torch.concatenate([ids_d, ids_d, ids_s_rand])
#         # feats_r = torch.concatenate([feats_q,feats_p])
#         feats_c = torch.concatenate([feats_q,feats_p,feats_rand])
#         fdist_mat = self.eucdist_computer(feats_c,feats_c)
#         # fdist_mat_np =  fdist_mat.detach().cpu().numpy()
#
#         # hard minning
#         N = len(ids)
#         is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_q.shape[0]*2]
#         is_pos_rand = torch.eye(fdist_mat.shape[0],dtype=torch.int)[feats_q.shape[0]*2:].bool()
#         is_pos = torch.concatenate([is_pos,is_pos_rand],dim=0)
#         pos_mat = fdist_mat.clone()
#         neg_mat = fdist_mat.clone()
#         pos_mat[~is_pos] = self.neg_ignore #just keep the value of positive unchanged, others are -inf
#         neg_mat[is_pos] = self.pos_ignore #just keep the value of positive unchanged, others are +inf
#         dist_hardp,relative_p_inds = torch.max(pos_mat, dim=-1)  # find the maximum distance of positive value in every row
#         dist_hardn,relative_n_inds = torch.min(neg_mat, dim=-1)  # find the minimum distance of negative value in every row
#
#         loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn)  )) #(dist_hardp - dist_hardn) large -> loss large
#         loss_p = loss[:feats_p.shape[0]*2].mean()
#         loss_rand = loss[feats_p.shape[0]*2:].mean()
#
#         return loss_p+loss_rand


# version 1:
# class WeightedSoftTripletLoss(nn.Module):
#     # writed by zwk
#     def __init__(self,alpha=3, dtype=torch.float32):
#         super(WeightedSoftTripletLoss, self).__init__()
#         self.alpha = alpha
#         self.pos_ignore = torch.tensor(pos_inf(dtype),dtype=dtype)
#         self.neg_ignore = torch.tensor(neg_inf(dtype),dtype=dtype)
#         self.eucdist_computer = LpDistance(normalize_embeddings=False) #metric = l2
#
#     def forward(self, feat_mat, pos_mask, dist_mat=None, feats_q=None, feats_p=None, feats_rand=None):
#         neg_mask = ~pos_mask
#         pos_mat = feat_mat.clone()
#         neg_mat = feat_mat.clone()
#
#         pos_mat[neg_mask] = self.neg_ignore #just keep the value of positive unchanged, others are -inf
#         neg_mat[pos_mask] = self.pos_ignore #just keep the value of positive unchanged, others are +inf
#         pos_most_hard_per_row,relative_p_inds = torch.max(pos_mat, dim=-1)  # find the maximum distance of positive value in every row
#         neg_most_hard_per_row,relative_n_inds = torch.min(neg_mat, dim=-1)  # find the minimum distance of negative value in every row
#         loss = torch.log(1 + torch.exp(self.alpha * (pos_most_hard_per_row - neg_most_hard_per_row))).mean()

        #debug
        # ids_d = torch.arange(feats_q.shape[0])
        # ids_s_rand = torch.arange(feats_rand.shape[0]) + ids_d.shape[0] * 2
        # ids = torch.concatenate([ids_d, ids_d, ids_s_rand])
        # feats_r = torch.concatenate([feats_q,feats_p])
        # feats_c = torch.concatenate([feats_q,feats_p,feats_rand])
        #
        # fdist_mat = self.eucdist_computer(feats_r, feats_c)
        # N = len(ids)
        # is_pos = ids.expand(N, N).eq(ids.expand(N, N).t())[:feats_r.shape[0]]
        # dist_hardp, relative_p_inds = torch.max(fdist_mat[is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        # dist_hardn, relative_n_inds = torch.min(fdist_mat[~is_pos].contiguous().view(feats_r.shape[0], -1), 1, keepdim=True)
        # loss = torch.log(1 + torch.exp(self.alpha  * (dist_hardp - dist_hardn))).mean() #(dist_hardp - dist_hardn) large -> loss large

        # return loss

