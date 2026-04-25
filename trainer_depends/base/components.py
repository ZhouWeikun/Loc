#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络组件工厂

统一管理所有网络组件的创建，避免重复代码
"""

import torch
import torch.nn as nn


class NetworkComponents:
    """
    网络组件工厂

    负责创建训练所需的所有网络组件：
    - 视觉编码器 (Backbone)
    - 特征聚合器 (Aggregator)
    - 位置编码器 (Positional Encoder)
    - Grid (INGP)
    - 条件MLP (Grid MLP)
    """

    def __init__(self, opt, device):
        """
        初始化组件工厂

        Args:
            opt: 配置对象
            device: torch.device对象
        """
        self.opt = opt
        self.device = device


    def create_visual_encoder(self):
        """
        创建视觉编码器 (Backbone)

        Returns:
            vis_encoder: 视觉编码器模型
        """
        from models.Backbone.util_mk_backbone import make_backbone

        backbone_name = self.opt.backbone
        backbone_config = getattr(self.opt, 'backbone_config', {})
        adapter_config = getattr(self.opt, 'adapter_config', {})
        vis_encoder = make_backbone(
            backbone_name,
            imgsize2net=getattr(self.opt, 'imgsize2net', 224),
            backbone_config=backbone_config,
            adapter_config=adapter_config,
        ).to(self.device)

        print(f"✅ 创建视觉编码器: {backbone_name}")
        print(f"   输出维度: {vis_encoder.output_channel}")
        if backbone_config:
            print(f"   backbone_config: {backbone_config}")
        if adapter_config:
            print(f"   adapter_config: {adapter_config}")

        return vis_encoder


    def create_aggregator(self, feat_dim):
        """
        创建特征聚合器

        Args:
            feat_dim: 输入特征维度

        Returns:
            aggregator: 聚合器模型
        """
        agg_type = getattr(self.opt, 'aggregator_type', 'salad')
        aggregator_config = dict(getattr(self.opt, 'aggregator_config', {}) or {})
        imgsize2net = int(getattr(self.opt, 'imgsize2net', 224))
        backbone_name = str(getattr(self.opt, 'backbone', '')).lower()
        patchsize = 14 if 'dinov2' in backbone_name else 16

        def _resolve_aggregator_output_dim(default=None):
            value = aggregator_config.get('output_dim', aggregator_config.get('out_channels', default))
            return int(value) if value is not None else None

        if agg_type == 'salad':
            from models.Head.salad_residual import SALAD_Residual

            use_residual = bool(aggregator_config.get('use_residual', True))
            cluster_dim = int(aggregator_config.get('cluster_dim', 64))
            num_clusters = aggregator_config.get('num_clusters')
            output_dim = _resolve_aggregator_output_dim()
            if output_dim is not None:
                residual_out_dim = output_dim if use_residual else (output_dim - feat_dim)
                if residual_out_dim <= 0:
                    raise ValueError(
                        f"SALAD output_dim must be larger than base_dim={feat_dim} when use_residual={use_residual}, "
                        f"got output_dim={output_dim}"
                    )
                if num_clusters is None:
                    if residual_out_dim % cluster_dim != 0:
                        raise ValueError(
                            f"SALAD output_dim={output_dim} is incompatible with cluster_dim={cluster_dim} "
                            f"and use_residual={use_residual}"
                        )
                    num_clusters = residual_out_dim // cluster_dim
                else:
                    num_clusters = int(num_clusters)
                    if num_clusters * cluster_dim != residual_out_dim:
                        raise ValueError(
                            f"SALAD num_clusters * cluster_dim = {num_clusters * cluster_dim} does not match "
                            f"required residual_out_dim={residual_out_dim} for output_dim={output_dim}"
                        )
            else:
                num_clusters = int(num_clusters) if num_clusters is not None else 16

            aggregator = SALAD_Residual(
                input_feat_dim=feat_dim,
                base_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                num_clusters=num_clusters,
                cluster_dim=cluster_dim,
                dropout=float(aggregator_config.get('dropout', 0.3)),
                with_dustbin=bool(aggregator_config.get('with_dustbin', True)),
                hidden_layer_dim=int(aggregator_config.get('hidden_layer_dim', 512)),
                use_residual=use_residual,
            ).to(self.device)
            print(f"✅ 创建SALAD聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(
                f"   SALAD配置: num_clusters={num_clusters}, cluster_dim={cluster_dim}, "
                f"hidden_layer_dim={int(aggregator_config.get('hidden_layer_dim', 512))}, "
                f"use_residual={use_residual}, with_dustbin={bool(aggregator_config.get('with_dustbin', True))}"
            )

        elif agg_type == 'gem':
            from models.Head.token_vpr_aggregators import TokenGeM

            output_dim = _resolve_aggregator_output_dim(feat_dim)
            p = float(aggregator_config.get('p', 3.0))
            eps = float(aggregator_config.get('eps', 1e-6))
            aggregator = TokenGeM(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                p=p,
                eps=eps,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建GeM聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(f"   GeM配置: output_dim={aggregator.output_dim}, p={p}, eps={eps}")

        elif agg_type == 'fsra':
            from models.Head.token_vpr_aggregators import TokenFSRA

            output_dim = _resolve_aggregator_output_dim()
            block = int(aggregator_config.get('block', 3))
            num_bottleneck = int(aggregator_config.get('num_bottleneck', 256))
            droprate = float(aggregator_config.get('droprate', 0.0))
            fuse_mode = str(aggregator_config.get('fuse_mode', 'concat'))
            use_cls_token = bool(aggregator_config.get('use_cls_token', True))
            aggregator = TokenFSRA(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                block=block,
                num_bottleneck=num_bottleneck,
                droprate=droprate,
                fuse_mode=fuse_mode,
                use_cls_token=use_cls_token,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建FSRA聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(
                f"   FSRA配置: output_dim={aggregator.output_dim}, block={block}, "
                f"num_bottleneck={num_bottleneck}, droprate={droprate}, "
                f"fuse_mode={fuse_mode}, use_cls_token={use_cls_token}"
            )

        elif agg_type == 'lpn':
            from models.Head.token_vpr_aggregators import TokenLPN

            output_dim = _resolve_aggregator_output_dim()
            block = int(aggregator_config.get('block', 3))
            num_bottleneck = int(aggregator_config.get('num_bottleneck', 256))
            droprate = float(aggregator_config.get('droprate', 0.0))
            fuse_mode = str(aggregator_config.get('fuse_mode', 'concat'))
            use_cls_token = bool(aggregator_config.get('use_cls_token', True))
            aggregator = TokenLPN(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                block=block,
                num_bottleneck=num_bottleneck,
                droprate=droprate,
                fuse_mode=fuse_mode,
                use_cls_token=use_cls_token,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建LPN聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(
                f"   LPN配置: output_dim={aggregator.output_dim}, block={block}, "
                f"num_bottleneck={num_bottleneck}, droprate={droprate}, "
                f"fuse_mode={fuse_mode}, use_cls_token={use_cls_token}"
            )

        elif agg_type in {'g2m', 'g2m_scalar_p'}:
            from models.Head.token_vpr_aggregators import TokenG2MScalarP

            out_channels = _resolve_aggregator_output_dim(feat_dim)
            rank = int(aggregator_config.get('rank', 1024))
            p = float(aggregator_config.get('p', 3.0))
            eps = float(aggregator_config.get('eps', 1e-6))
            aggregator = TokenG2MScalarP(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                output_dim=out_channels,
                rank=rank,
                p=p,
                eps=eps,
            ).to(self.device)
            print(f"✅ 创建G2M聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(f"   G2M配置: output_dim={out_channels}, rank={rank}, p={p}, eps={eps}")

        elif agg_type == 'g2m_channelwise_p':
            from models.Head.token_vpr_aggregators import TokenG2MChannelwiseP
            out_channels = _resolve_aggregator_output_dim(feat_dim)
            rank = int(aggregator_config.get('rank', 1024))
            p = float(aggregator_config.get('p', 3.0))
            eps = float(aggregator_config.get('eps', 1e-6))
            aggregator = TokenG2MChannelwiseP(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                output_dim=out_channels,
                rank=rank,
                p=p,
                eps=eps,
            ).to(self.device)
            print(f"✅ 创建G2M聚合器 (channelwise p)")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(f"   G2M配置: output_dim={out_channels}, rank={rank}, p={p}, eps={eps}")

        elif agg_type == 'netvlad':
            from models.Head.token_vpr_aggregators import TokenNetVLAD

            output_dim = _resolve_aggregator_output_dim()
            num_clusters = int(aggregator_config.get('num_clusters', 16))
            alpha = float(aggregator_config.get('alpha', 100.0))
            normalize_input = bool(aggregator_config.get('normalize_input', True))
            aggregator = TokenNetVLAD(
                input_feat_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                num_clusters=num_clusters,
                alpha=alpha,
                normalize_input=normalize_input,
                output_dim=output_dim,
            ).to(self.device)
            print("✅ 创建NetVLAD聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
            print(
                f"   NetVLAD配置: num_clusters={num_clusters}, alpha={alpha}, "
                f"normalize_input={normalize_input}"
            )

        else:
            raise ValueError(f"未知的聚合器类型: {agg_type}")

        if not hasattr(aggregator, 'output_dim'):
            aggregator.output_dim = feat_dim
        print(f"   输入维度: {feat_dim}D")
        print(f"   输出维度: {int(aggregator.output_dim)}D")

        return aggregator


    def create_positional_encoders(self, multires_rc=8, multires_rot=6, multires_scale=4):
        """
        创建位置编码器

        Args:
            multires_rc: xy坐标的多分辨率级数
            multires_rot: 旋转的多分辨率级数
            multires_scale: 尺度的多分辨率级数

        Returns:
            dict: 包含三个编码器的字典
                  {'rc': encoder, 'rot': encoder, 'scale': encoder}
        """
        from models.pos_encoder import PositionalEncoder

        encoders = {
            'rc': PositionalEncoder(
                input_dims=2,
                include_input=True,
                multires=multires_rc
            ),
            'rot': PositionalEncoder(
                input_dims=2,
                include_input=True,
                multires=multires_rot
            ),
            'scale': PositionalEncoder(
                input_dims=1,
                include_input=True,
                multires=multires_scale
            )
        }

        total_dim = sum(enc.out_dim for enc in encoders.values())
        print(f"✅ 创建位置编码器:")
        print(f"   rc:    {encoders['rc'].out_dim}D (multires={multires_rc})")
        print(f"   rot:   {encoders['rot'].out_dim}D (multires={multires_rot})")
        print(f"   scale: {encoders['scale'].out_dim}D (multires={multires_scale})")
        print(f"   总维度: {total_dim}D")

        return encoders


    def create_coords_5d_encoder(self, multires_rc=8, multires_rot=6, multires_scale=4):
        """
        创建5D坐标编码器 (新版本)

        Args:
            multires_rc: rc坐标（row, col）的多分辨率级数
            multires_rot: 旋转的多分辨率级数
            multires_scale: 尺度的多分辨率级数

        Returns:
            encoder: Coords5DEncoder实例
        """
        from models.pos_encoder_euc5d import Coords5DEncoder

        encoder = Coords5DEncoder(
            multires_rc=multires_rc,
            multires_rot=multires_rot,
            multires_scale=multires_scale,
            include_input_rc_scale=True,
            log_sampling=True
        ).to(self.device)

        print(f"✅ 创建5D坐标编码器:")
        print(f"   rc:    2D → {encoder.rc_encoder.out_dim}D (multires={multires_rc})")
        print(f"   rot:   2D → {encoder.rot_encoder.out_dim}D (multires={multires_rot})")
        print(f"   scale: 1D → {encoder.scale_encoder.out_dim}D (multires={multires_scale})")
        print(f"   总维度: {encoder.out_dim}D")

        return encoder


    def create_grid(self, config_path=None):
        """
        创建Grid (INGP)

        Args:
            config_path: Grid配置文件路径，如果为None则使用默认路径

        Returns:
            grid: Grid对象
        """
        from app.nerf.main_nerf import NeRFAppConfig
        from wisp.config._tyro import parse_args_tyro_v1
        from wisp.config import instantiate
        import sys

        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if config_path is None:
            config_path = getattr(self.opt, 'p_grid_config_yaml', None)

        if not config_path:
            config_path = os.path.join(project_root, 'trainer_depends/configs/nerf_hash.yaml')
        elif not os.path.isabs(str(config_path)):
            config_path = os.path.join(project_root, str(config_path))

        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Grid config YAML not found: {config_path}")

        # 临时保存并清空 sys.argv，避免与 tyro 的参数解析冲突
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # 只保留程序名

        try:
            grid_args = parse_args_tyro_v1(NeRFAppConfig, config_path)
        finally:
            # 恢复原始 sys.argv
            sys.argv = original_argv

        blas = instantiate(grid_args.blas, pointcloud=None)
        grid = instantiate(grid_args.grid, blas=blas).to(self.device)

        grid_base_dim = int(getattr(grid, 'feature_dim', 0))
        grid_num_lods = int(getattr(grid, 'num_lods', len(getattr(grid, 'resolutions', [])) or 1))
        grid_multiscale_type = str(getattr(grid, 'multiscale_type', 'sum'))
        grid_output_dim = grid_base_dim * grid_num_lods if grid_multiscale_type == 'cat' else grid_base_dim
        grid.output_dim = grid_output_dim

        print(f"✅ 创建Grid (INGP)")
        print(f"   配置文件: {config_path}")
        print(f"   单层特征维度: {grid_base_dim}")
        print(f"   LOD数量: {grid_num_lods}")
        print(f"   多尺度聚合: {grid_multiscale_type}")
        print(f"   有效输出维度: {grid_output_dim}")

        # 保存grid_args供后续使用
        self.grid_args = grid_args

        return grid


    def create_hash_lod_aggregator(self, config=None):
        """
        创建 HashGrid LOD 聚合器。

        YAML 只需要暴露核心参数；实现细节使用模块默认值。
        """
        if config is None:
            config = getattr(self.opt, 'hash_lod_aggregator', None)
        config = dict(config or {})
        if not bool(config.get('enabled', False)):
            return None

        mode = str(config.get('mode', 'scale_softmax'))
        if mode != 'scale_softmax':
            raise ValueError(f"Unsupported hash_lod_aggregator.mode: {mode}")

        from models.hash_lod_aggregator import ScaleLodAggregator

        aggregator = ScaleLodAggregator(
            num_lods=int(config.get('num_lods', 4)),
            per_lod_dim=int(config.get('per_lod_dim', 1024)),
            output_dim=int(config.get('output_dim', config.get('per_lod_dim', 1024))),
            coord_source=str(config.get('coord_source', 'scale')),
            scale_source=str(config.get('scale_source', 'norm_log_scale')),
        ).to(self.device)

        print("✅ 创建HashGrid LOD聚合器")
        print(f"   模式: {mode}")
        print(f"   条件源: {aggregator.coord_source}")
        print(f"   scale源: {aggregator.scale_source}")
        print(f"   LOD数量: {aggregator.num_lods}")
        print(f"   单LOD维度: {aggregator.per_lod_dim}D")
        print(f"   输入维度: {aggregator.input_dim}D")
        print(f"   输出维度: {aggregator.output_dim}D")

        return aggregator


    def create_grid_mlp(self, input_dim, condition_dim, hidden_dim=512, num_blocks=1, output_dim=None):
        """
        创建条件MLP (Grid MLP)

        Args:
            input_dim: 输入特征维度
            condition_dim: 条件特征维度（位置编码维度）
            hidden_dim: 隐藏层维度
            num_blocks: MLP块数量
            output_dim: 输出特征维度；为None时与input_dim相同

        Returns:
            grid_mlp: 条件MLP模型
        """
        from models.cond_modulator_shallow_serial import SerialModulatorShallow

        if output_dim is None:
            output_dim = input_dim

        grid_mlp = SerialModulatorShallow(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            condition_operator='add'
        ).to(self.device)

        print(f"✅ 创建条件MLP:")
        print(f"   输入维度: {input_dim}D")
        print(f"   条件维度: {condition_dim}D")
        print(f"   隐藏维度: {hidden_dim}D")
        print(f"   块数量: {num_blocks}")
        print(f"   输出维度: {output_dim}D")

        return grid_mlp


    def create_metric_net(self,
                          feat_dim=1024,
                          coord_dim=128,
                          branch_hidden_dim=768,
                          branch_output_dim=512,
                          resblock_hidden_dim=384,
                          resblock_output_dim=256,
                          dropout=0.1,
                          init_weights=True,
                          output_activation=None,
                          ):
        """
        创建MetricNet

        Args:
            feat_dim: 输入特征维度（默认1024）
            coord_dim: 坐标编码维度
            branch_hidden_dim: Branch MLP的隐藏层维度
            branch_output_dim: Branch MLP的输出维度（调制前）
            resblock_hidden_dim: 残差块的隐藏层维度
            resblock_output_dim: 残差块的输出维度（调制后）
            dropout: Dropout率
            init_weights: 是否使用自定义权重初始化

        Returns:
            metric_net: MetricNet模型
        """
        from models.metric_net import MetricNet

        metric_net = MetricNet(
            feat_dim=feat_dim,
            coord_dim=coord_dim,
            branch_hidden_dim=branch_hidden_dim,
            branch_output_dim=branch_output_dim,
            resblock_hidden_dim=resblock_hidden_dim,
            resblock_output_dim=resblock_output_dim,
            dropout=dropout,
            init_weights=init_weights,
            output_activation=output_activation
        ).to(self.device)

        print(f"✅ 创建MetricNet:")
        print(f"   特征维度: {feat_dim}D")
        print(f"   坐标维度: {coord_dim}D")
        print(f"   Branch隐藏维度: {branch_hidden_dim}D")
        print(f"   Branch输出维度: {branch_output_dim}D")
        print(f"   Resblock隐藏维度: {resblock_hidden_dim}D")
        print(f"   Resblock输出维度: {resblock_output_dim}D")
        print(f"   Dropout: {dropout}")
        print(f"   自定义初始化: {init_weights}")

        return metric_net
