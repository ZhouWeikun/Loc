import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1InfoNCELoss(nn.Module):
    """InfoNCE for Stage-1 UAV query to satellite reference matching."""

    def __init__(self, temperature=0.1, negative_mode="batch_and_explicit"):
        super().__init__()
        self.temperature = float(temperature)
        self.negative_mode = str(negative_mode).lower()
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.negative_mode not in {"batch", "explicit", "batch_and_explicit"}:
            raise ValueError(
                f"Unsupported negative_mode={negative_mode!r}. "
                "Expected one of: batch, explicit, batch_and_explicit."
            )

    def forward(self, query, positive_key, explicit_negative_keys=None):
        if query.dim() != 2 or positive_key.dim() != 2:
            raise ValueError("query and positive_key must be [N, D].")
        if query.shape != positive_key.shape:
            raise ValueError(
                f"query and positive_key must have the same shape, got {query.shape} and {positive_key.shape}"
            )
        if explicit_negative_keys is not None and explicit_negative_keys.dim() != 2:
            raise ValueError("explicit_negative_keys must be [M, D] when provided.")

        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        pos_logits = torch.sum(query * positive_key, dim=-1, keepdim=True)

        neg_logits = []
        if self.negative_mode in {"batch", "batch_and_explicit"} and query.shape[0] > 1:
            batch_logits = query @ positive_key.transpose(0, 1)
            self_mask = torch.eye(query.shape[0], device=query.device, dtype=torch.bool)
            batch_neg_logits = batch_logits.masked_select(~self_mask).view(query.shape[0], -1)
            neg_logits.append(batch_neg_logits)

        if (
            self.negative_mode in {"explicit", "batch_and_explicit"}
            and explicit_negative_keys is not None
            and explicit_negative_keys.shape[0] > 0
        ):
            explicit_negative_keys = F.normalize(explicit_negative_keys, dim=-1)
            neg_logits.append(query @ explicit_negative_keys.transpose(0, 1))

        if not neg_logits:
            raise ValueError("InfoNCE needs at least one negative sample.")

        logits = torch.cat([pos_logits] + neg_logits, dim=1) / self.temperature
        labels = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)
