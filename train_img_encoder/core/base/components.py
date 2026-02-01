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
        vis_encoder = make_backbone(backbone_name).to(self.device)

        print(f"✅ 创建视觉编码器: {backbone_name}")
        print(f"   输出维度: {vis_encoder.output_channel}")

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

        if agg_type == 'salad':
            from models.Head.salad_residual import SALAD_Residual
            aggregator = SALAD_Residual(
                input_feat_dim=feat_dim,
                base_dim=feat_dim,
                patchsize=16,
                num_clusters=16,
                cluster_dim=64
            ).to(self.device)
            print(f"✅ 创建SALAD聚合器")

        elif agg_type == 'g2m':
            from models.Head.G2M import G2M
            aggregator = G2M(
                in_channels=feat_dim,
                out_channels=feat_dim,
                rank=1024
            ).to(self.device)
            print(f"✅ 创建G2M聚合器")

        else:
            raise ValueError(f"未知的聚合器类型: {agg_type}")

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


    def create_coords_5d_encoder(self, multires_xy=10, multires_rot=4, multires_scale=4):
        """
        创建5D坐标编码器 (新版本)

        Args:
            multires_xy: xy坐标的多分辨率级数
            multires_rot: 旋转的多分辨率级数
            multires_scale: 尺度的多分辨率级数

        Returns:
            encoder: Coords5DEncoder实例
        """
        from models.pos_encoder import Coords5DEncoder

        encoder = Coords5DEncoder(
            multires_xy=multires_xy,
            multires_rot=multires_rot,
            multires_scale=multires_scale,
            include_input=True,
            log_sampling=True
        )

        print(f"✅ 创建5D坐标编码器:")
        print(f"   xy:    2D → {encoder.xy_encoder.out_dim}D (multires={multires_xy})")
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

        if config_path is None:
            config_path = '/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/nerf_hash.yaml'

        grid_args = parse_args_tyro_v1(NeRFAppConfig, config_path)

        blas = instantiate(grid_args.blas, pointcloud=None)
        grid = instantiate(grid_args.grid, blas=blas).to(self.device)

        print(f"✅ 创建Grid (INGP)")
        print(f"   配置文件: {config_path}")
        print(f"   特征维度: {grid.feature_dim}")

        # 保存grid_args供后续使用
        self.grid_args = grid_args

        return grid


    def create_grid_mlp(self, input_dim, condition_dim, hidden_dim=512, num_blocks=1):
        """
        创建条件MLP (Grid MLP)

        Args:
            input_dim: 输入特征维度
            condition_dim: 条件特征维度（位置编码维度）
            hidden_dim: 隐藏层维度
            num_blocks: MLP块数量

        Returns:
            grid_mlp: 条件MLP模型
        """
        from models.cond_modulator_shallow_serial import SerialModulatorShallow

        grid_mlp = SerialModulatorShallow(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=input_dim,
            condition_operator='add'
        ).to(self.device)

        print(f"✅ 创建条件MLP:")
        print(f"   输入维度: {input_dim}D")
        print(f"   条件维度: {condition_dim}D")
        print(f"   隐藏维度: {hidden_dim}D")
        print(f"   块数量: {num_blocks}")

        return grid_mlp


    def create_metric_net(self, query_dim, ref_dim, coord_dim, hidden_dim=256, num_layers=3):
        """
        创建MetricNet

        Args:
            query_dim: query特征维度
            ref_dim: reference特征维度
            coord_dim: 坐标编码维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数

        Returns:
            metric_net: MetricNet模型
        """
        from models.metric_net import MetricNet

        metric_net = MetricNet(
            query_dim=query_dim,
            ref_dim=ref_dim,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        print(f"✅ 创建MetricNet:")
        print(f"   query维度: {query_dim}D")
        print(f"   ref维度: {ref_dim}D")
        print(f"   coord维度: {coord_dim}D")
        print(f"   隐藏维度: {hidden_dim}D")
        print(f"   层数: {num_layers}")

        return metric_net
