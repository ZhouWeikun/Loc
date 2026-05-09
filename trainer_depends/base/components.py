#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal network component factory for the three-stage INGP pipeline."""

import os
import sys


class NetworkComponents:
    """Create only the modules used by the minimal Stage1/2/3 entrypoints."""

    def __init__(self, opt, device):
        self.opt = opt
        self.device = device

    def create_visual_encoder(self):
        from models.Backbone.util_mk_backbone import make_backbone

        backbone_name = self.opt.backbone
        backbone_config = getattr(self.opt, "backbone_config", {})
        adapter_config = getattr(self.opt, "adapter_config", {})
        vis_encoder = make_backbone(
            backbone_name,
            imgsize2net=getattr(self.opt, "imgsize2net", 224),
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
        agg_type = str(getattr(self.opt, "aggregator_type", "salad")).lower()
        if agg_type != "salad":
            raise ValueError(f"Minimal pipeline only supports aggregator_type='salad', got {agg_type!r}")

        from models.Head.salad_residual import SALAD_Residual

        aggregator_config = dict(getattr(self.opt, "aggregator_config", {}) or {})
        imgsize2net = int(getattr(self.opt, "imgsize2net", 224))
        backbone_name = str(getattr(self.opt, "backbone", "")).lower()
        patchsize = 14 if "dinov2" in backbone_name else 16

        use_residual = bool(aggregator_config.get("use_residual", True))
        cluster_dim = int(aggregator_config.get("cluster_dim", 64))
        num_clusters = aggregator_config.get("num_clusters")
        output_dim = aggregator_config.get("output_dim", aggregator_config.get("out_channels", None))
        output_dim = int(output_dim) if output_dim is not None else None

        if output_dim is not None:
            residual_out_dim = output_dim if use_residual else (output_dim - feat_dim)
            if residual_out_dim <= 0:
                raise ValueError(
                    f"SALAD output_dim must be larger than base_dim={feat_dim} "
                    f"when use_residual={use_residual}, got output_dim={output_dim}"
                )
            if num_clusters is None:
                if residual_out_dim % cluster_dim != 0:
                    raise ValueError(
                        f"SALAD output_dim={output_dim} is incompatible with "
                        f"cluster_dim={cluster_dim} and use_residual={use_residual}"
                    )
                num_clusters = residual_out_dim // cluster_dim
            else:
                num_clusters = int(num_clusters)
                if num_clusters * cluster_dim != residual_out_dim:
                    raise ValueError(
                        f"SALAD num_clusters * cluster_dim = {num_clusters * cluster_dim} "
                        f"does not match required residual_out_dim={residual_out_dim}"
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
            dropout=float(aggregator_config.get("dropout", 0.3)),
            with_dustbin=bool(aggregator_config.get("with_dustbin", True)),
            hidden_layer_dim=int(aggregator_config.get("hidden_layer_dim", 512)),
            use_residual=use_residual,
        ).to(self.device)

        print("✅ 创建SALAD聚合器")
        print(f"   token_grid: {imgsize2net // patchsize}x{imgsize2net // patchsize} (patchsize={patchsize})")
        print(f"   输入维度: {feat_dim}D")
        print(f"   输出维度: {int(aggregator.output_dim)}D")
        return aggregator

    def create_coords_5d_encoder(self, multires_rc=8, multires_rot=6, multires_scale=4):
        from models.pos_encoder_euc5d import Coords5DEncoder

        encoder = Coords5DEncoder(
            multires_rc=multires_rc,
            multires_rot=multires_rot,
            multires_scale=multires_scale,
            include_input_rc_scale=True,
            log_sampling=True,
        ).to(self.device)

        print("✅ 创建5D坐标编码器:")
        print(f"   rc:    2D → {encoder.rc_encoder.out_dim}D (multires={multires_rc})")
        print(f"   rot:   2D → {encoder.rot_encoder.out_dim}D (multires={multires_rot})")
        print(f"   scale: 1D → {encoder.scale_encoder.out_dim}D (multires={multires_scale})")
        print(f"   总维度: {encoder.out_dim}D")
        return encoder

    def create_grid(self, config_path=None):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wisp_root = os.environ.get(
            "KAOLIN_WISP_ROOT",
            os.path.join(os.path.dirname(project_root), "cache", "kaolin-wisp"),
        )
        if os.path.isdir(wisp_root) and wisp_root not in sys.path:
            sys.path.insert(0, wisp_root)

        from app.nerf.main_nerf import NeRFAppConfig
        from wisp.config import instantiate
        from wisp.config._tyro import parse_args_tyro_v1

        if config_path is None:
            config_path = getattr(self.opt, "p_grid_config_yaml", None)
        if not config_path:
            config_path = os.path.join(project_root, "trainer_depends/configs/nerf_hash_wingtra.yaml")
        elif not os.path.isabs(str(config_path)):
            config_path = os.path.join(project_root, str(config_path))

        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Grid config YAML not found: {config_path}")

        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]
        try:
            grid_args = parse_args_tyro_v1(NeRFAppConfig, config_path)
        finally:
            sys.argv = original_argv

        blas = instantiate(grid_args.blas, pointcloud=None)
        grid = instantiate(grid_args.grid, blas=blas).to(self.device)

        grid_base_dim = int(getattr(grid, "feature_dim", 0))
        grid_num_lods = int(getattr(grid, "num_lods", len(getattr(grid, "resolutions", [])) or 1))
        grid_multiscale_type = str(getattr(grid, "multiscale_type", "sum"))
        grid.output_dim = grid_base_dim * grid_num_lods if grid_multiscale_type == "cat" else grid_base_dim

        print("✅ 创建Grid (INGP)")
        print(f"   配置文件: {config_path}")
        print(f"   单层特征维度: {grid_base_dim}")
        print(f"   LOD数量: {grid_num_lods}")
        print(f"   多尺度聚合: {grid_multiscale_type}")
        print(f"   有效输出维度: {grid.output_dim}")
        self.grid_args = grid_args
        return grid

    def create_hash_lod_aggregator(self, config=None):
        if config is None:
            config = getattr(self.opt, "hash_lod_aggregator", None)
        config = dict(config or {})
        if not bool(config.get("enabled", False)):
            return None

        mode = str(config.get("mode", "scale_softmax"))
        if mode != "scale_softmax":
            raise ValueError(f"Unsupported hash_lod_aggregator.mode: {mode}")

        from models.hash_lod_aggregator import ScaleLodAggregator

        aggregator = ScaleLodAggregator(
            num_lods=int(config.get("num_lods", 4)),
            per_lod_dim=int(config.get("per_lod_dim", 1024)),
            output_dim=int(config.get("output_dim", config.get("per_lod_dim", 1024))),
            coord_source=str(config.get("coord_source", "scale")),
            scale_source=str(config.get("scale_source", "norm_log_scale")),
        ).to(self.device)

        print("✅ 创建HashGrid LOD聚合器")
        print(f"   模式: {mode}")
        print(f"   输入维度: {aggregator.input_dim}D")
        print(f"   输出维度: {aggregator.output_dim}D")
        return aggregator

    def create_grid_mlp(self, input_dim, condition_dim, hidden_dim=512, num_blocks=1, output_dim=None):
        from models.stage2_grid_mlp import Stage2GridFeatureMLP

        if output_dim is None:
            output_dim = input_dim

        arch = str(getattr(self.opt, "grid_mlp_arch", "residual_cond")).lower()
        if arch != "residual_cond":
            raise ValueError(f"Minimal pipeline only supports grid_mlp_arch='residual_cond', got {arch!r}")

        grid_mlp = Stage2GridFeatureMLP(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            arch=arch,
            use_coord_condition=bool(getattr(self.opt, "grid_mlp_use_coord_condition", True)),
        ).to(self.device)

        print("✅ 创建Grid MLP:")
        print(f"   输入维度: {input_dim}D")
        print(f"   条件维度: {condition_dim}D")
        print(f"   隐藏维度: {hidden_dim}D")
        print(f"   块数量: {num_blocks}")
        print(f"   输出维度: {output_dim}D")
        return grid_mlp
