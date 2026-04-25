"""
Token-friendly GeM / G2M / FSRA / LPN / NetVLAD aggregators for Stage-1 control experiments.

Reference implementations checked against:
- GeM: filipradenovic/cnnimageretrieval-pytorch
- NetVLAD: Nanne/pytorch-NetVlad

These wrappers adapt ViT-style token outputs [B, T, C] to the [B, C, H, W]
feature-map interface expected by the original pooling operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenAggregatorBase(nn.Module):
    def __init__(self, input_feat_dim: int, img_hw=(224, 224), patchsize: int = 14) -> None:
        super().__init__()
        self.input_feat_dim = int(input_feat_dim)
        self.img_hw = tuple(int(v) for v in img_hw)
        self.patchsize = int(patchsize)
        self.token_hw = (self.img_hw[0] // self.patchsize, self.img_hw[1] // self.patchsize)

    def _tokens_to_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != self.input_feat_dim:
                raise ValueError(
                    f"Expected feature-map channels={self.input_feat_dim}, got {x.shape[1]}"
                )
            return x

        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C] or [B, C, H, W], got shape={tuple(x.shape)}")

        bsz, num_tokens, feat_dim = x.shape
        if feat_dim != self.input_feat_dim:
            raise ValueError(f"Expected token dim={self.input_feat_dim}, got {feat_dim}")

        expected_patch_tokens = self.token_hw[0] * self.token_hw[1]
        if num_tokens < expected_patch_tokens:
            raise ValueError(
                f"Expected at least {expected_patch_tokens} tokens for token_hw={self.token_hw}, got {num_tokens}"
            )

        # Keep the last patch tokens. This handles CLS / register tokens prepended
        # by ViT-family backbones while staying strict about the spatial token count.
        patch_tokens = x[:, -expected_patch_tokens:, :]
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()
        return patch_tokens.view(bsz, feat_dim, self.token_hw[0], self.token_hw[1])


class TokenGeM(nn.Module):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        p: float = 3.0,
        eps: float = 1e-6,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        from models.Head.G2M_scalar_p import GeM

        self.backbone = _TokenAggregatorBase(
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
        )
        self.pool = GeM(p=p, eps=eps)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_feat_dim)
        self.proj = (
            nn.Linear(int(input_feat_dim), self.output_dim)
            if self.output_dim != int(input_feat_dim)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone._tokens_to_feature_map(x)
        desc = self.pool(fmap).flatten(1)
        if self.proj is not None:
            desc = self.proj(desc)
        return F.normalize(desc, p=2, dim=1)


class TokenNetVLAD(nn.Module):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        num_clusters: int = 16,
        alpha: float = 100.0,
        normalize_input: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = _TokenAggregatorBase(
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
        )
        self.num_clusters = int(num_clusters)
        self.dim = int(input_feat_dim)
        self.alpha = float(alpha)
        self.normalize_input = bool(normalize_input)

        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        self._init_params()

        self.vlad_dim = self.num_clusters * self.dim
        self.output_dim = int(output_dim) if output_dim is not None else self.vlad_dim
        self.proj = nn.Linear(self.vlad_dim, self.output_dim) if self.output_dim != self.vlad_dim else None

    def _init_params(self) -> None:
        with torch.no_grad():
            self.conv.weight.copy_(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            if self.conv.bias is not None:
                self.conv.bias.copy_(-self.alpha * self.centroids.norm(dim=1))

    def initialize_centroids(self, centroids: torch.Tensor, alpha: float | None = None) -> None:
        centroids = torch.as_tensor(centroids, dtype=self.centroids.dtype, device=self.centroids.device)
        if centroids.shape != (self.num_clusters, self.dim):
            raise ValueError(
                f"Expected centroids with shape {(self.num_clusters, self.dim)}, got {tuple(centroids.shape)}"
            )
        if alpha is not None:
            self.alpha = float(alpha)
        with torch.no_grad():
            self.centroids.copy_(centroids)
        self._init_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone._tokens_to_feature_map(x)
        n, c = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x).view(n, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(n, c, -1)

        # Loop over clusters to keep memory bounded for large ViT descriptors.
        vlad = torch.zeros(
            (n, self.num_clusters, c),
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        for cluster_idx in range(self.num_clusters):
            centroid = self.centroids[cluster_idx : cluster_idx + 1, :].unsqueeze(-1)
            residual = x_flatten - centroid
            residual = residual * soft_assign[:, cluster_idx : cluster_idx + 1, :]
            vlad[:, cluster_idx : cluster_idx + 1, :] = residual.sum(dim=-1, keepdim=True).transpose(1, 2)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.reshape(n, -1)
        if self.proj is not None:
            vlad = self.proj(vlad)
        return F.normalize(vlad, p=2, dim=1)


class TokenFSRA(nn.Module):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        block: int = 3,
        num_bottleneck: int = 256,
        droprate: float = 0.0,
        fuse_mode: str = "concat",
        use_cls_token: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        from models.Head.FSRA import FSRA_wo_CLS

        self.backbone = _TokenAggregatorBase(
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
        )
        self.block = max(1, int(block))
        self.num_bottleneck = int(num_bottleneck)
        self.droprate = float(droprate)
        self.fuse_mode = str(fuse_mode).strip().lower()
        self.use_cls_token = bool(use_cls_token)

        if self.fuse_mode not in {"concat", "mean", "global"}:
            raise ValueError(
                "TokenFSRA fuse_mode must be one of ('concat', 'mean', 'global'), "
                f"got {fuse_mode}"
            )

        fsra_opt = SimpleNamespace(
            droprate=self.droprate,
            in_planes=int(input_feat_dim),
            num_bottleneck=self.num_bottleneck,
            block=self.block,
            w_classify=False,
        )
        self.pool = FSRA_wo_CLS(fsra_opt)

        if self.fuse_mode == "concat" and self.block > 1:
            raw_output_dim = self.num_bottleneck * (1 + self.block)
        else:
            raw_output_dim = self.num_bottleneck
        self.output_dim = int(output_dim) if output_dim is not None else int(raw_output_dim)
        self.proj = (
            nn.Linear(int(raw_output_dim), self.output_dim)
            if self.output_dim != int(raw_output_dim)
            else None
        )

    def _tokens_to_fsra_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            fmap = self.backbone._tokens_to_feature_map(x)
            patch_tokens = fmap.flatten(2).transpose(1, 2).contiguous()
            global_token = patch_tokens.mean(dim=1, keepdim=True)
            return torch.cat([global_token, patch_tokens], dim=1)

        if x.ndim != 3:
            raise ValueError(f"TokenFSRA expects [B, T, C] or [B, C, H, W], got shape={tuple(x.shape)}")

        bsz, num_tokens, feat_dim = x.shape
        if feat_dim != self.backbone.input_feat_dim:
            raise ValueError(f"Expected token dim={self.backbone.input_feat_dim}, got {feat_dim}")

        expected_patch_tokens = self.backbone.token_hw[0] * self.backbone.token_hw[1]
        if num_tokens < expected_patch_tokens:
            raise ValueError(
                f"Expected at least {expected_patch_tokens} tokens for token_hw={self.backbone.token_hw}, got {num_tokens}"
            )

        patch_tokens = x[:, -expected_patch_tokens:, :]
        if self.use_cls_token and num_tokens > expected_patch_tokens:
            global_token = x[:, :1, :]
        else:
            global_token = patch_tokens.mean(dim=1, keepdim=True)
        return torch.cat([global_token, patch_tokens], dim=1).reshape(
            bsz, 1 + expected_patch_tokens, feat_dim
        )

    def _fuse_branches(self, features):
        if isinstance(features, torch.Tensor):
            if features.ndim == 2:
                return features
            if features.ndim == 3:
                if self.fuse_mode == "concat":
                    return features.flatten(1)
                if self.fuse_mode == "mean":
                    return features.mean(dim=-1)
                return features[..., 0]
            raise ValueError(f"Unexpected FSRA tensor output shape: {tuple(features.shape)}")

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError("Unexpected FSRA output type; expected non-empty list/tuple or tensor.")

        branch_feats = [feat if feat.ndim == 2 else feat.reshape(feat.shape[0], -1) for feat in features]
        if self.fuse_mode == "concat":
            return torch.cat(branch_feats, dim=1)
        stacked = torch.stack(branch_feats, dim=-1)
        if self.fuse_mode == "mean":
            return stacked.mean(dim=-1)
        return stacked[..., 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fsra_input = self._tokens_to_fsra_input(x)
        desc = self._fuse_branches(self.pool(fsra_input))
        if self.proj is not None:
            desc = self.proj(desc)
        return F.normalize(desc, p=2, dim=1)


class TokenLPN(nn.Module):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        block: int = 3,
        num_bottleneck: int = 256,
        droprate: float = 0.0,
        fuse_mode: str = "concat",
        use_cls_token: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        from models.Head.LPN import LPN

        self.backbone = _TokenAggregatorBase(
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
        )
        self.block = max(1, int(block))
        self.num_bottleneck = int(num_bottleneck)
        self.droprate = float(droprate)
        self.fuse_mode = str(fuse_mode).strip().lower()
        self.use_cls_token = bool(use_cls_token)

        if self.fuse_mode not in {"concat", "mean", "global"}:
            raise ValueError(
                "TokenLPN fuse_mode must be one of ('concat', 'mean', 'global'), "
                f"got {fuse_mode}"
            )

        lpn_opt = SimpleNamespace(
            block=self.block,
            in_planes=int(input_feat_dim),
            nclasses=1,
            droprate=self.droprate,
            num_bottleneck=self.num_bottleneck,
        )
        self.pool = LPN(lpn_opt)

        raw_output_dim = self.num_bottleneck * (1 + self.block) if self.fuse_mode == "concat" else self.num_bottleneck
        self.output_dim = int(output_dim) if output_dim is not None else int(raw_output_dim)
        self.proj = (
            nn.Linear(int(raw_output_dim), self.output_dim)
            if self.output_dim != int(raw_output_dim)
            else None
        )

    def _tokens_to_lpn_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            fmap = self.backbone._tokens_to_feature_map(x)
            patch_tokens = fmap.flatten(2).transpose(1, 2).contiguous()
            global_token = patch_tokens.mean(dim=1, keepdim=True)
            return torch.cat([global_token, patch_tokens], dim=1)

        if x.ndim != 3:
            raise ValueError(f"TokenLPN expects [B, T, C] or [B, C, H, W], got shape={tuple(x.shape)}")

        bsz, num_tokens, feat_dim = x.shape
        if feat_dim != self.backbone.input_feat_dim:
            raise ValueError(f"Expected token dim={self.backbone.input_feat_dim}, got {feat_dim}")

        expected_patch_tokens = self.backbone.token_hw[0] * self.backbone.token_hw[1]
        if num_tokens < expected_patch_tokens:
            raise ValueError(
                f"Expected at least {expected_patch_tokens} tokens for token_hw={self.backbone.token_hw}, got {num_tokens}"
            )

        patch_tokens = x[:, -expected_patch_tokens:, :]
        if self.use_cls_token and num_tokens > expected_patch_tokens:
            global_token = x[:, :1, :]
        else:
            global_token = patch_tokens.mean(dim=1, keepdim=True)
        return torch.cat([global_token, patch_tokens], dim=1).reshape(
            bsz, 1 + expected_patch_tokens, feat_dim
        )

    def _extract_lpn_features(self, outputs):
        if (
            isinstance(outputs, (list, tuple))
            and len(outputs) == 2
            and isinstance(outputs[0], (list, tuple))
        ):
            return outputs[1]
        return outputs

    def _fuse_branches(self, features):
        if isinstance(features, torch.Tensor):
            if features.ndim == 2:
                return features
            if features.ndim == 3:
                if self.fuse_mode == "concat":
                    return features.flatten(1)
                if self.fuse_mode == "mean":
                    return features.mean(dim=-1)
                return features[..., 0]
            raise ValueError(f"Unexpected LPN tensor output shape: {tuple(features.shape)}")

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError("Unexpected LPN output type; expected non-empty list/tuple or tensor.")

        branch_feats = [feat if feat.ndim == 2 else feat.reshape(feat.shape[0], -1) for feat in features]
        if self.fuse_mode == "concat":
            return torch.cat(branch_feats, dim=1)
        stacked = torch.stack(branch_feats, dim=-1)
        if self.fuse_mode == "mean":
            return stacked.mean(dim=-1)
        return stacked[..., 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lpn_input = self._tokens_to_lpn_input(x)
        desc = self._fuse_branches(self._extract_lpn_features(self.pool(lpn_input)))
        if self.proj is not None:
            desc = self.proj(desc)
        return F.normalize(desc, p=2, dim=1)


class _TokenG2MBase(nn.Module):
    def __init__(
        self,
        g2m_cls,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        output_dim: int | None = None,
        rank: int = 64,
        p: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.backbone = _TokenAggregatorBase(
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
        )
        self.output_dim = int(output_dim) if output_dim is not None else int(input_feat_dim)
        self.pool = g2m_cls(
            in_channels=int(input_feat_dim),
            out_channels=self.output_dim,
            rank=int(rank),
            p=float(p),
            eps=float(eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone._tokens_to_feature_map(x)
        return self.pool(fmap)


class TokenG2MScalarP(_TokenG2MBase):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        output_dim: int | None = None,
        rank: int = 64,
        p: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        from models.Head.G2M_scalar_p import G2M as G2MScalarP

        super().__init__(
            g2m_cls=G2MScalarP,
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
            output_dim=output_dim,
            rank=rank,
            p=p,
            eps=eps,
        )


class TokenG2MChannelwiseP(_TokenG2MBase):
    def __init__(
        self,
        input_feat_dim: int,
        img_hw=(224, 224),
        patchsize: int = 14,
        output_dim: int | None = None,
        rank: int = 64,
        p: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        from models.Head.G2M_channelwise_p import G2M as G2MChannelwiseP

        super().__init__(
            g2m_cls=G2MChannelwiseP,
            input_feat_dim=input_feat_dim,
            img_hw=img_hw,
            patchsize=patchsize,
            output_dim=output_dim,
            rank=rank,
            p=p,
            eps=eps,
        )


__all__ = ["TokenGeM", "TokenG2MScalarP", "TokenG2MChannelwiseP", "TokenFSRA", "TokenLPN", "TokenNetVLAD"]
