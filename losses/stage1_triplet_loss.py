import torch
from torch import nn
import torch.nn.functional as F
import math

def pos_inf(dtype):
    return torch.finfo(dtype).max

def neg_inf(dtype):
    return torch.finfo(dtype).min


###############################################################################################

class tripletLoss_singleEdge_hardest_fm_mask(nn.Module):
    def __init__(self, dtype=torch.float32, slope_pos=5,slope_neg=5,decoupling=False, **kwargs):
        super(tripletLoss_singleEdge_hardest_fm_mask, self).__init__()
        self.register_buffer("pos_ignore", torch.tensor(pos_inf(dtype), dtype=dtype))
        self.register_buffer("neg_ignore", torch.tensor(neg_inf(dtype), dtype=dtype))
        self.alpha = slope_pos
        self.beta = slope_neg
        self.decoupling = decoupling
        self.weight_dist_func = lambda x:1/(1+torch.exp(-8.5*(x-0.15)))

    def forward(self, feat_mat,
                pos_mask,
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
        else:
            raise NotImplementedError("This function has not been implemented yet, baby is looking forward to it!")
