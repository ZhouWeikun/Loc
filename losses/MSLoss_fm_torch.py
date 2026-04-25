"""
Minimal torch-only implementation of the official Multi-Similarity loss used by SALAD.

Adapted from the MIT-licensed implementations in:
- KevinMusgrave/pytorch-metric-learning
- serizba/salad

This file intentionally avoids importing the full pytorch-metric-learning package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pos_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).max


def _neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def _masked_logsumexp(
    x: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
    add_one: bool = True,
    dim: int = 1,
) -> torch.Tensor:
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, _neg_inf(x.dtype))
    if add_one:
        pad_shape = list(x.shape)
        pad_shape[dim] = 1
        zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=dim)
    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


def _get_matches_and_diffs(
    labels: torch.Tensor,
    ref_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    same_source = ref_labels is None or ref_labels is labels
    if ref_labels is None:
        ref_labels = labels
    labels = labels.reshape(-1)
    ref_labels = ref_labels.reshape(-1)
    matches = labels.unsqueeze(1).eq(ref_labels.unsqueeze(0))
    diffs = ~matches
    if same_source and matches.shape[0] == matches.shape[1]:
        matches.fill_diagonal_(False)
        diffs.fill_diagonal_(False)
    return matches, diffs


def get_all_pairs_indices(
    labels: torch.Tensor,
    ref_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    matches, diffs = _get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


class CosineSimilarity:
    is_inverted = True
    normalize_embeddings = True

    def __call__(
        self,
        query_emb: torch.Tensor,
        ref_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_emb is None:
            ref_emb = query_emb
        query_emb = F.normalize(query_emb, p=2, dim=1)
        ref_emb = F.normalize(ref_emb, p=2, dim=1)
        return torch.matmul(query_emb, ref_emb.transpose(-1, -2))

    def margin(self, x: torch.Tensor | float, y: torch.Tensor | float):
        return y - x


class MultiSimilarityMinerTorch:
    def __init__(
        self,
        epsilon: float = 0.1,
        distance: Optional[CosineSimilarity] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.distance = distance if distance is not None else CosineSimilarity()

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ref_emb: Optional[torch.Tensor] = None,
        ref_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mine(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)

    def mine(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ref_emb: Optional[torch.Tensor] = None,
        ref_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if ref_emb is None:
            ref_emb = embeddings
        if ref_labels is None:
            ref_labels = labels

        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = get_all_pairs_indices(labels, ref_labels)

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.empty(0, device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        pos_ignore = _pos_inf(mat.dtype) if self.distance.is_inverted else _neg_inf(mat.dtype)
        neg_ignore = _neg_inf(mat.dtype) if self.distance.is_inverted else _pos_inf(mat.dtype)

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore

        if ref_emb is embeddings and mat.shape[0] == mat.shape[1]:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1))
            hard_neg_idx = torch.where(neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1))
        else:
            hard_pos_idx = torch.where(pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1))
            hard_neg_idx = torch.where(neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1))

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]
        return a1, p, a2, n


class MultiSimilarityLossTorch(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 50.0,
        base: float = 0.0,
        distance: Optional[CosineSimilarity] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.base = float(base)
        self.distance = distance if distance is not None else CosineSimilarity()
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        indices_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ref_emb: Optional[torch.Tensor] = None,
        ref_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_emb is None:
            ref_emb = embeddings
        if ref_labels is None:
            ref_labels = labels

        sim_mat = self.distance(embeddings, ref_emb)
        if indices_tuple is None:
            a1, p, a2, n = get_all_pairs_indices(labels, ref_labels)
        else:
            if len(indices_tuple) != 4:
                raise ValueError("indices_tuple must be a 4-tuple: (a1, p, a2, n)")
            a1, p, a2, n = indices_tuple

        pos_mask = torch.zeros_like(sim_mat, dtype=torch.bool)
        neg_mask = torch.zeros_like(sim_mat, dtype=torch.bool)
        if len(a1) > 0:
            pos_mask[a1, p] = True
        if len(a2) > 0:
            neg_mask[a2, n] = True
        return self.compute_loss_from_similarity_matrix(sim_mat, pos_mask, neg_mask)

    def compute_loss_from_similarity_matrix(
        self,
        sim_mat: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        pos_exp = self.distance.margin(sim_mat, self.base)
        neg_exp = self.distance.margin(self.base, sim_mat)

        pos_loss = (1.0 / self.alpha) * _masked_logsumexp(
            self.alpha * pos_exp,
            keep_mask=pos_mask.bool(),
            add_one=True,
            dim=1,
        )
        neg_loss = (1.0 / self.beta) * _masked_logsumexp(
            self.beta * neg_exp,
            keep_mask=neg_mask.bool(),
            add_one=True,
            dim=1,
        )

        loss = (pos_loss + neg_loss).squeeze(1)
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


MSLoss_fm_torch = MultiSimilarityLossTorch


__all__ = [
    "CosineSimilarity",
    "MultiSimilarityLossTorch",
    "MultiSimilarityMinerTorch",
    "MSLoss_fm_torch",
    "get_all_pairs_indices",
]
