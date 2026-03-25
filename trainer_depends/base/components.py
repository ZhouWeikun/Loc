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
        vis_encoder = make_backbone(
            backbone_name,
            imgsize2net=getattr(self.opt, 'imgsize2net', 224),
            backbone_config=backbone_config,
        ).to(self.device)

        print(f"✅ 创建视觉编码器: {backbone_name}")
        print(f"   输出维度: {vis_encoder.output_channel}")
        if backbone_config:
            print(f"   backbone_config: {backbone_config}")

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
        imgsize2net = int(getattr(self.opt, 'imgsize2net', 224))
        backbone_name = str(getattr(self.opt, 'backbone', '')).lower()
        patchsize = 14 if 'dinov2' in backbone_name else 16

        if agg_type == 'salad':
            from models.Head.salad_residual import SALAD_Residual
            aggregator = SALAD_Residual(
                input_feat_dim=feat_dim,
                base_dim=feat_dim,
                img_hw=(imgsize2net, imgsize2net),
                patchsize=patchsize,
                num_clusters=16,
                cluster_dim=64
            ).to(self.device)
            print(f"✅ 创建SALAD聚合器")
            print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")

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
