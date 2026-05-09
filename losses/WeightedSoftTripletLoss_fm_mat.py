import torch
from torch import nn


def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min


class MSLoss_fm_mat(nn.Module):
    def __init__(self, dtype=torch.float32, slope_pos=7.5, slope_neg=7.5, tau=0.5, **kwargs):
        super().__init__()
        self.pos_ignore = torch.tensor(pos_inf(dtype), dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype), dtype=dtype)
        self.alpha = slope_pos
        self.beta = slope_neg
        self.tau = tau

    def forward(self, feat_mat, pos_mask, metric="dist"):
        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()

        if metric != "dist":
            raise NotImplementedError("Minimal MSLoss_fm_mat only supports metric='dist'.")

        pos_mat[neg_mask] = self.neg_ignore
        neg_mat[pos_mask] = self.pos_ignore

        pos_most_hard_per_row = torch.max(pos_mat, dim=-1)
        neg_most_hard_per_row = torch.min(neg_mat, dim=-1)
        hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask
        hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask

        hard_pos = feat_mat.masked_fill(~hard_pos_mask, self.neg_ignore)
        hard_pos = self.alpha * (hard_pos - self.tau)
        hard_pos = torch.cat([hard_pos, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
        loss_pos = 1 / self.alpha * torch.logsumexp(hard_pos, dim=-1, keepdim=True).mean()

        hard_neg = feat_mat.masked_fill(~hard_neg_mask, self.pos_ignore)
        hard_neg = -self.beta * (hard_neg - self.tau)
        hard_neg = torch.cat([hard_neg, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
        loss_neg = 1 / self.beta * torch.logsumexp(hard_neg, dim=-1, keepdim=True).mean()
        return loss_pos + loss_neg


class SWTLoss_fm_mat(nn.Module):
    def __init__(self, dtype=torch.float32, slope_pos=5, slope_neg=5, decoupling=False, **kwargs):
        super().__init__()
        self.pos_ignore = torch.tensor(pos_inf(dtype), dtype=dtype)
        self.neg_ignore = torch.tensor(neg_inf(dtype), dtype=dtype)
        self.alpha = slope_pos
        self.beta = slope_neg
        self.decoupling = decoupling
        self.weight_dist_func = lambda x: 1 / (1 + torch.exp(-8.5 * (x - 0.15)))

    def forward(self, feat_mat, pos_mask, dist_mat=None, metric="dist", w_weight=False):
        if metric != "dist":
            raise NotImplementedError("Minimal SWTLoss_fm_mat only supports metric='dist'.")

        neg_mask = ~pos_mask
        pos_mat = feat_mat.clone()
        neg_mat = feat_mat.clone()
        pos_mat[neg_mask] = self.neg_ignore
        neg_mat[pos_mask] = self.pos_ignore

        pos_most_hard_per_row = torch.max(pos_mat, dim=-1)
        neg_most_hard_per_row = torch.min(neg_mat, dim=-1)

        if not self.decoupling:
            base_loss = torch.log(
                1 + torch.exp(self.alpha * (pos_most_hard_per_row[0] - neg_most_hard_per_row[0]))
            )
            if not w_weight:
                return base_loss.mean()

            indices_hard_pos = pos_most_hard_per_row[1].unsqueeze(1)
            dist_pos_most_hard_per_row = torch.gather(dist_mat, dim=1, index=indices_hard_pos)
            indices_hard_neg = neg_most_hard_per_row[1].unsqueeze(1)
            dist_neg_most_hard_per_row = torch.gather(dist_mat, dim=1, index=indices_hard_neg)
            neg_weight = self.weight_dist_func(dist_neg_most_hard_per_row - dist_pos_most_hard_per_row).squeeze()
            return (neg_weight * base_loss).mean()

        hard_pos_mask = torch.gt(feat_mat, neg_most_hard_per_row[0].unsqueeze(1)) * pos_mask
        hard_neg_mask = torch.lt(feat_mat, pos_most_hard_per_row[0].unsqueeze(1)) * neg_mask

        delta_pos = feat_mat - neg_most_hard_per_row[0].unsqueeze(1)
        hard_pos = delta_pos.masked_fill(~hard_pos_mask, self.neg_ignore)
        hard_pos = self.alpha * hard_pos
        hard_pos = torch.cat([hard_pos, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
        loss_per_row_pos = torch.logsumexp(hard_pos, dim=-1)
        non_zero_mask_pos = loss_per_row_pos > 0
        loss_pos = loss_per_row_pos[non_zero_mask_pos].mean() if non_zero_mask_pos.sum() > 0 else torch.tensor(0.0, device=feat_mat.device)

        delta_neg = pos_most_hard_per_row[0].unsqueeze(1) - feat_mat
        hard_neg = delta_neg.masked_fill(~hard_neg_mask, self.pos_ignore)
        hard_neg = -self.beta * hard_neg
        hard_neg = torch.cat([hard_neg, torch.zeros((feat_mat.shape[0], 1), device=feat_mat.device)], dim=-1)
        loss_per_row_neg = torch.logsumexp(hard_neg, dim=-1)
        non_zero_mask_neg = loss_per_row_neg > 0
        loss_neg = loss_per_row_neg[non_zero_mask_neg].mean() if non_zero_mask_neg.sum() > 0 else torch.tensor(0.0, device=feat_mat.device)
        return loss_neg + loss_pos
