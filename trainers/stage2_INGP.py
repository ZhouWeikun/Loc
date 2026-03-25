#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2: Grid HashFit Trainer

训练目标：
- grid (INGP)
- grid_mlp (条件调制器)

前置条件：
- 需要Stage 1训练好的 vis_encoder + vis_aggregator（冻结使用）

训练策略：
- 使用MSE Loss将Grid特征拟合到视觉特征
- 使用4D坐标的位置编码作为条件
"""

import torch
import torch.nn.functional as TF
import tqdm
import time
import sys
import os
import yaml
from collections.abc import Sequence

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trainer_depends.base.trainer_base import BaseTrainer
from trainer_depends.base.components import NetworkComponents
# from trainer_depends.utils.util_udf_computer import UDFComputer
from models.pos_encoder import encode_4d_coords
from trainers.util_stage2_gallery_manager import (
    Stage2ReferenceGalleryBank,
    Stage2ReferenceGalleryFeatureConfig,
    Stage2ReferenceGalleryLayoutConfig,
)
from trainers.util_stage2_retrieval_evaluator import (
    Stage2RetrievalEvalConfig,
    Stage2RetrievalEvaluator,
)
from trainer_depends.config.parser import get_parse, print_config_summary


STAGE1_INHERIT_SCOPES = ('network', 'data', 'scenes', 'hardware')
STAGE1_INHERIT_SECTION_MAP = {
    'network': 'network_setting',
    'data': 'data_setting',
    'hardware': 'hardware_setting',
}
STAGE1_INHERIT_DATA_KEYS = (
    'pad_mode',
    'imgsize2net',
    'satimgsize2crop',
    'n_rand2sample_per_pos',
    'split_train_ratio',
    'split_mode',
)
STAGE1_INHERIT_NETWORK_KEYS = (
    'backbone',
    'freeze_backbone',
    'backbone_config',
    'aggregator_type',
    'aggregator_config',
)
STAGE1_INHERIT_KEY_WHITELIST = {
    'data_setting': STAGE1_INHERIT_DATA_KEYS,
    'network_setting': STAGE1_INHERIT_NETWORK_KEYS,
}


class GridHashFitTrainer(BaseTrainer):
    """
    Stage 2: Grid HashFit Trainer

    训练Grid拟合视觉特征场
    """

    def __init__(self, opt=None):
        """初始化Stage 2 Trainer"""
        should_print_final_config = (
            (opt is None)
            or bool(getattr(opt, 'inherit_stage1_yaml', ''))
            or not bool(getattr(opt, '_config_summary_printed', False))
        )
        if opt is None:
            opt = get_parse(print_summary=False)
        opt = self._apply_inherit_stage1_yaml(opt)
        if should_print_final_config:
            print_config_summary(opt, header="最终生效配置:")

        super().__init__(opt)

        # 初始化网络组件
        self._init_networks()

        # 预加载 Stage 1 权重，供训练与测试共用 query 视觉特征分支。
        if self.opt.load_stage1_ckpt:
            self._load_stage1_checkpoint()

        # 设置可训练参数
        self._setup_trainable_params()

    @staticmethod
    def _normalize_inherit_stage1_scope(scope_value):
        if not scope_value:
            scopes = ['network', 'data']
        elif isinstance(scope_value, str):
            scopes = [part.strip() for part in scope_value.split(',') if part.strip()]
        elif isinstance(scope_value, Sequence):
            scopes = []
            for item in scope_value:
                if isinstance(item, str):
                    scopes.extend(part.strip() for part in item.split(',') if part.strip())
                else:
                    scopes.append(str(item).strip())
        else:
            scopes = [str(scope_value).strip()]

        invalid = [scope for scope in scopes if scope not in STAGE1_INHERIT_SCOPES]
        if invalid:
            raise ValueError(
                f"Unsupported inherit_stage1_scope: {invalid}. "
                f"Supported scopes: {STAGE1_INHERIT_SCOPES}"
            )

        deduped = []
        for scope in scopes:
            if scope not in deduped:
                deduped.append(scope)
        return tuple(deduped or ('network', 'data'))

    @staticmethod
    def _apply_inherit_stage1_yaml(opt):
        """
        在Stage 2初始化网络前，从指定的Stage 1 YAML/opts中按scope继承参数。
        对 data/network 采用白名单，只补充 Stage 2 当前 YAML 未显式声明的 Stage 1 相关配置；
        scenes/hardware 若显式请求，则整段继承。
        """
        inherit_yaml = getattr(opt, 'inherit_stage1_yaml', '')
        if not inherit_yaml:
            return opt

        inherit_yaml = os.path.abspath(inherit_yaml)
        if not os.path.exists(inherit_yaml):
            raise FileNotFoundError(f"inherit_stage1_yaml not found: {inherit_yaml}")
        inherit_scopes = GridHashFitTrainer._normalize_inherit_stage1_scope(
            getattr(opt, 'inherit_stage1_scope', 'network,data')
        )

        with open(inherit_yaml, 'r', encoding='utf-8') as f:
            stage1_cfg = yaml.safe_load(f) or {}

        explicit_stage2_keys = GridHashFitTrainer._collect_stage2_explicit_keys(
            getattr(opt, 'p_yaml', '')
        )
        inherited_summary = {}
        skipped_summary = {}
        for scope in inherit_scopes:
            if scope == 'scenes':
                if 'scenes_setting' in explicit_stage2_keys:
                    skipped_summary['scenes_setting'] = ['<explicit_in_stage2_yaml>']
                    continue
                scenes_cfg = stage1_cfg.get('scenes_setting')
                if scenes_cfg:
                    setattr(opt, 'scenes_setting', scenes_cfg)
                    inherited_summary['scenes_setting'] = list(scenes_cfg.keys())
                continue

            section_name = STAGE1_INHERIT_SECTION_MAP[scope]
            section_cfg = stage1_cfg.get(section_name, {})
            if not isinstance(section_cfg, dict):
                continue

            allowed_keys = STAGE1_INHERIT_KEY_WHITELIST.get(section_name, None)
            explicit_keys = explicit_stage2_keys.get(section_name, set())
            inherited_keys = []
            skipped_keys = []
            for key, value in section_cfg.items():
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                if key in explicit_keys:
                    skipped_keys.append(key)
                    continue
                setattr(opt, key, value)
                inherited_keys.append(key)
            if inherited_keys:
                inherited_summary[section_name] = inherited_keys
            if skipped_keys:
                skipped_summary[section_name] = skipped_keys

        if inherited_summary:
            print(f"✅ 从Stage 1配置继承参数: {inherit_yaml}")
            print(f"   scopes: {', '.join(inherit_scopes)}")
            for section_name, keys in inherited_summary.items():
                print(f"   {section_name}: {', '.join(keys)}")
        else:
            print(f"⚠️  inherit_stage1_yaml未提供可继承的scope字段: {inherit_yaml}")
        for section_name, keys in skipped_summary.items():
            print(f"   skip override by stage2 yaml | {section_name}: {', '.join(keys)}")

        opt.inherit_stage1_yaml = inherit_yaml
        opt.inherit_stage1_scope = ','.join(inherit_scopes)
        return opt

    @staticmethod
    def _load_yaml_dict(yaml_path):
        if not yaml_path:
            return {}
        yaml_path = str(yaml_path).strip()
        if not yaml_path:
            return {}
        yaml_path_abs = yaml_path if os.path.isabs(yaml_path) else os.path.join(project_root, yaml_path)
        if not os.path.exists(yaml_path_abs):
            return {}
        with open(yaml_path_abs, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}

    @staticmethod
    def _merge_section_key_sets(target, source):
        for section_name, keys in source.items():
            target.setdefault(section_name, set()).update(keys)
        return target

    @staticmethod
    def _collect_declared_keys_from_cfg(cfg):
        declared = {}
        for section_name in ('data_setting', 'network_setting', 'hardware_setting'):
            section_cfg = cfg.get(section_name, None)
            if isinstance(section_cfg, dict):
                declared[section_name] = set(section_cfg.keys())
        if isinstance(cfg.get('scenes_setting', None), dict):
            declared['scenes_setting'] = {'__section__'}
        return declared

    @classmethod
    def _collect_stage2_explicit_keys(cls, yaml_path, _visited=None):
        yaml_path = str(yaml_path or '').strip()
        if not yaml_path:
            return {}
        yaml_path_abs = yaml_path if os.path.isabs(yaml_path) else os.path.join(project_root, yaml_path)
        if _visited is None:
            _visited = set()
        if yaml_path_abs in _visited:
            return {}
        _visited.add(yaml_path_abs)

        cfg = cls._load_yaml_dict(yaml_path_abs)
        if not cfg:
            return {}

        declared = {}
        base_yaml = cfg.get('p_yaml') or cfg.get('exp_setting', {}).get('p_yaml')
        if base_yaml:
            declared = cls._collect_stage2_explicit_keys(base_yaml, _visited=_visited)
        return cls._merge_section_key_sets(declared, cls._collect_declared_keys_from_cfg(cfg))

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"


    def _init_networks(self):
        """初始化所有网络组件"""
        print("\n" + "="*80)
        print("初始化 Stage 2 网络组件")
        print("="*80)

        components = NetworkComponents(self.opt, self.device)

        # Stage 1组件（将被冻结）
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel
        self.vis_aggregator = components.create_aggregator(
            self.feat_patch_dim
        )
        self.feat_q_dim = int(getattr(self.vis_aggregator, 'output_dim', self.feat_patch_dim))

        posenc_multires_rc = int(getattr(self.opt, 'posenc_multires_rc', 8))
        posenc_multires_rot = int(getattr(self.opt, 'posenc_multires_rot', 6))
        posenc_multires_scale = int(getattr(self.opt, 'posenc_multires_scale', 4))

        # 位置编码器
        # version1:
        self.pos_encoder_864 = components.create_coords_5d_encoder(
            multires_rc=posenc_multires_rc,
            multires_rot=posenc_multires_rot,
            multires_scale=posenc_multires_scale
        )
        self.pos_encoder_grid = self.pos_encoder_864

        grid_mlp_hidden_dim = int(getattr(self.opt, 'grid_mlp_hidden_dim', 512))
        grid_mlp_num_blocks = int(getattr(self.opt, 'grid_mlp_num_blocks', 1))

        # Stage 2组件（将被训练）
        self.grid = components.create_grid()
        self.feat_grid_dim = int(getattr(self.grid, 'output_dim', self.feat_q_dim))
        # version1：
        self.grid_mlp = components.create_grid_mlp(
            self.feat_grid_dim,
            self.pos_encoder_grid.out_dim,
            hidden_dim=grid_mlp_hidden_dim,
            num_blocks=grid_mlp_num_blocks,
            output_dim=self.feat_q_dim,
        )

        # 保存grid_args供后续使用
        self.grid_args = components.grid_args

        print("="*80 + "\n")


    def _load_stage1_checkpoint(self):
        """加载Stage 1训练好的模型"""
        print(f"\n加载Stage 1 checkpoint: {self.opt.load_stage1_ckpt}")

        self._load_checkpoint(
            self.opt.load_stage1_ckpt,
            {
                'vis_encoder': self.vis_encoder,
                'vis_aggregator': self.vis_aggregator
            }
        )

        print("✅ Stage 1模型加载完成\n")


    def _setup_trainable_params(self):
        """设置可训练参数"""
        # 冻结Stage 1组件
        for module in [self.vis_encoder, self.vis_aggregator]:
            for param in module.parameters():
                param.requires_grad = False

        # 训练Stage 2组件
        self.param2optimize = {
            'grid': self.grid,
            'grid_mlp': self.grid_mlp
        }

        self.param2freeze = {
            'vis_encoder': self.vis_encoder,
            'vis_aggregator': self.vis_aggregator
        }

        # 动态生成参数配置信息
        trainable_names = ', '.join(self.param2optimize.keys())
        frozen_names = ', '.join(self.param2freeze.keys())

        print("参数配置:")
        print(f"  可训练: {trainable_names}")
        print(f"  冻结:   {frozen_names}\n")


    def _make_train_checkpoint_modules(self):
        """
        构造训练态 checkpoint 需要保存/恢复的对象。
        AMP 开启时额外包含 GradScaler 状态。
        """
        modules = dict(self.param2optimize)
        if getattr(self.opt, 'autocast', False) and hasattr(self, 'scaler') and self.scaler is not None:
            modules['amp_scaler'] = self.scaler
        return modules


    def _get_feats_fm_grid(self, grid_coords_normed, z_padding=0.025, compress_to_unit_interval=False):
        """
        纯粹的 Grid 查表函数

        Args:
            grid_coords_normed: [N, 3] or [B, N, 3]
                                对应 (nr, nc, z_axis)，z_axis=log_scale or nrot
                                范围必须是 [-1, 1] (来自 CoordsNormProcessor)
            z_padding: float, Z轴边界留空比例 (0.0 ~ 0.5).
                       例如 0.05 表示最终送入 HashGrid 的 Z 轴有效范围保留在内部 90% 区间。
                       这对 Z=Rotation 方案至关重要，防止边界断裂问题。
            compress_to_unit_interval: 是否在进入 Wisp HashGrid 前，先手动将坐标从 [-1, 1]
                       压缩到 [0, 1]。默认 False。
        Returns:
            feats_grid: [N, feat_dim] or [B, N, feat_dim]
        """
        # 1. 维度展平 (Handle Batch)
        input_shape = grid_coords_normed.shape
        if len(input_shape) == 3:
            coords_flat = grid_coords_normed.flatten(0, 1)  # [B*N, 3]
        else:
            coords_flat = grid_coords_normed

        # 2. 根据配置选择是否手动压缩到 [0, 1]。
        # 当前 Wisp HashGrid 内部会自行做 [-1, 1] -> [0, 1] 映射，因此默认保持 False。
        if compress_to_unit_interval:
            grid_input = (coords_flat + 1.0) * 0.5
            if z_padding > 0.0:
                scale_factor = 1.0 - 2.0 * z_padding
                grid_input[:, 2] = grid_input[:, 2] * scale_factor + z_padding
            grid_input = torch.clamp(grid_input, 0.0, 1.0)
        else:
            grid_input = coords_flat.clone()
            if z_padding > 0.0:
                # 在 [-1, 1] 空间中对称压缩，交给 Wisp 内部再映射到 [0, 1]。
                scale_factor = 1.0 - 2.0 * z_padding
                grid_input[:, 2] = grid_input[:, 2] * scale_factor
            grid_input = torch.clamp(grid_input, -1.0, 1.0)

        # 3. 查表 (自动处理多分辨率插值)
        feats_grid = self.grid.interpolate(grid_input, len(self.grid.active_lods) - 1) #拼接得到的多尺度特征

        # 4. 恢复维度
        if len(input_shape) == 3:
            feats_grid = feats_grid.view(input_shape[0], input_shape[1], -1)

        return feats_grid


    def _ensure_stage2_eval_runtime(self):
        """确保Stage 2评估所需的数据集和坐标归一化器已初始化"""
        if not hasattr(self, 'sat_dataset') or self.sat_dataset is None:
            self._init_datasets(create_train_loader=False)

        if not hasattr(self, 'coord_normer') or self.coord_normer is None:
            from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
            self.coord_normer = CoordsNormProcessor(self.sat_dataset)


    def _enter_model_eval_mode(self):
        """保存当前模式并切换所有相关模型到eval"""
        models_all = list(self.param2optimize.values()) + list(self.param2freeze.values())
        orig_modes = [model.training for model in models_all]
        for model in models_all:
            model.eval()
        return models_all, orig_modes


    @staticmethod
    def _restore_model_modes(models_all, orig_modes):
        """恢复模型原始train/eval状态"""
        for model, was_train in zip(models_all, orig_modes):
            model.train(was_train)


    def _extract_stage2_feats_from_coords_chunk(self, coords_4d, normalize=True):
        """
        Stage 2 gallery/query backend:
        4D坐标 -> coord_normer -> grid -> pos_encoder -> grid_mlp
        """
        self._ensure_stage2_eval_runtime()

        coords_4d = coords_4d.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            coords_6d = self.coord_normer.raw_to_norm(coords_4d, append_linear_rot=True)
            grid_coords_3d = torch.cat([coords_6d[:, 0:2], coords_6d[:, -1:]], dim=-1)
            feats_grid = self._get_feats_fm_grid(grid_coords_3d)
            coords_encoded = self.pos_encoder_grid(coords_6d[:, :5])
            feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_encoded)
            if normalize:
                feats_grid = TF.normalize(feats_grid, dim=-1)
        return feats_grid


    def build_stage2_gallery_bank(self, layout_cfg, feature_cfg=None):
        """构建Stage 2专用gallery bank"""
        self._ensure_stage2_eval_runtime()

        gallery_bank = Stage2ReferenceGalleryBank(sat_dataset=self.sat_dataset, trainer=self)
        gallery_bank.build_coords(layout_cfg)
        gallery_bank.build_features(feature_cfg)
        return gallery_bank


    def evaluate_stage2_gallery_bank(self, layout_cfg, eval_cfg=None, feature_cfg=None):
        """统一的Stage 2 gallery build + retrieval eval入口"""
        self._ensure_stage2_eval_runtime()

        models_all, orig_modes = self._enter_model_eval_mode()
        try:
            gallery_bank = self.build_stage2_gallery_bank(
                layout_cfg=layout_cfg,
                feature_cfg=feature_cfg,
            )
            retrieval_evaluator = Stage2RetrievalEvaluator(
                trainer=self,
                gallery_bank=gallery_bank,
                logger=self.logger,
            )
            return retrieval_evaluator.evaluate(eval_cfg=eval_cfg)
        finally:
            self._restore_model_modes(models_all, orig_modes)


    @staticmethod
    def _resolve_stage2_gallery_ckpt_tag(ckpt_path):
        if not ckpt_path:
            return None
        ckpt_name = os.path.splitext(os.path.basename(str(ckpt_path)))[0]
        return ckpt_name or None


    def resolve_gallery_bank_save_dir(self, layout_cfg, root_dir=None, name_prefix=None, ckpt_path=None):
        """
        Stage 2 gallery bank保存路径解析。
        结构上对齐Stage 1，但这里不区分scene_name参数，直接使用当前主scene。
        """
        self._ensure_stage2_eval_runtime()

        cfg = layout_cfg if isinstance(layout_cfg, Stage2ReferenceGalleryLayoutConfig) else (
            Stage2ReferenceGalleryLayoutConfig(**layout_cfg)
        )

        scene_name = getattr(self.sat_dataset, 'name', 'default_scene')
        root_dir = root_dir or os.path.join(project_root, "gen_fm_exps", "gallery_bank_stage2")
        name_prefix = name_prefix or scene_name

        overlap_tag = f"overlap{int(round(float(cfg.overlap) * 100.0)):03d}"
        layout_tags = [name_prefix, cfg.mode, overlap_tag]
        if cfg.fixed_scale is not None:
            layout_tags.append(f"fixs{float(cfg.fixed_scale):.3f}".replace('.', 'p'))
        if abs(float(cfg.fixed_rot)) > 1e-6 and cfg.mode in ('rc', 'rc_scale'):
            layout_tags.append(f"fixr{float(cfg.fixed_rot):.3f}".replace('.', 'p'))
        if cfg.mode in ('rc_rot', 'rc_rot_scale'):
            layout_tags.append(f"drot{float(cfg.delta_rot_deg):g}".replace('.', 'p'))
        if cfg.mode in ('rc_scale', 'rc_rot_scale'):
            layout_tags.append(f"nscale{int(cfg.n_scales)}")
        layout_tags.append(str(cfg.scale_mode))

        base_dir = os.path.join(root_dir, "_".join(layout_tags))
        ckpt_tag = self._resolve_stage2_gallery_ckpt_tag(
            ckpt_path or self._get_stage2_checkpoint_path()
        )
        if ckpt_tag:
            return os.path.join(base_dir, ckpt_tag)
        return base_dir


    def build_or_load_gallery_bank(
            self,
            layout_cfg,
            feature_cfg=None,
            gallery_save_dir=None,
            load_if_exists=True,
            save_gallery=True,
            init_datasets=True,
            load_ckpt=False,
            gallery_root_dir=None,
            gallery_name_prefix=None,
    ):
        """
        Stage 2版 gallery build/load orchestration。
        """
        if init_datasets or (not hasattr(self, 'sat_dataset')):
            self._ensure_stage2_eval_runtime()

        if load_ckpt:
            self._load_checkpoints_for_test()
        ckpt_path = self._get_stage2_checkpoint_path()

        if feature_cfg is None:
            feature_cfg = Stage2ReferenceGalleryFeatureConfig()
        elif not isinstance(feature_cfg, Stage2ReferenceGalleryFeatureConfig):
            feature_cfg = Stage2ReferenceGalleryFeatureConfig(**feature_cfg)

        if gallery_save_dir is None and save_gallery:
            gallery_save_dir = self.resolve_gallery_bank_save_dir(
                layout_cfg=layout_cfg,
                root_dir=gallery_root_dir,
                name_prefix=gallery_name_prefix,
                ckpt_path=ckpt_path,
            )

        coords_path = None if gallery_save_dir is None else os.path.join(gallery_save_dir, "coords_gallery.pt")
        can_load = bool(load_if_exists and coords_path and os.path.exists(coords_path))

        models_all, orig_modes = self._enter_model_eval_mode()
        try:
            if can_load:
                gallery_bank = Stage2ReferenceGalleryBank.load(
                    gallery_save_dir,
                    sat_dataset=self.sat_dataset,
                    trainer=self,
                    build_faiss=bool(feature_cfg.build_faiss),
                )
                if self.logger:
                    self.logger.info(f"[Stage2 Gallery Bank] loaded from {gallery_save_dir}")
                else:
                    print(f"[Stage2 Gallery Bank] loaded from {gallery_save_dir}")
                if gallery_bank.feats_gallery is None:
                    if self.logger:
                        self.logger.info("[Stage2 Gallery Bank] cached gallery has no features, rebuilding them.")
                    else:
                        print("[Stage2 Gallery Bank] cached gallery has no features, rebuilding them.")
                    gallery_bank.build_features(feature_cfg)
                    if gallery_save_dir is not None and save_gallery:
                        gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
            else:
                gallery_bank = self.build_stage2_gallery_bank(layout_cfg=layout_cfg, feature_cfg=feature_cfg)
                if self.logger:
                    self.logger.info(
                        f"[Stage2 Gallery Bank] n_points={gallery_bank.coords_gallery.shape[0]}, "
                        f"mode={gallery_bank.meta.get('mode', None)}"
                    )
                else:
                    print(
                        f"[Stage2 Gallery Bank] n_points={gallery_bank.coords_gallery.shape[0]}, "
                        f"mode={gallery_bank.meta.get('mode', None)}"
                    )
                if gallery_save_dir is not None and save_gallery:
                    gallery_bank.save(gallery_save_dir, save_feats=True, save_meta=True)
                    if self.logger:
                        self.logger.info(f"[Stage2 Gallery Bank] saved to {gallery_save_dir}")
                    else:
                        print(f"[Stage2 Gallery Bank] saved to {gallery_save_dir}")
        finally:
            self._restore_model_modes(models_all, orig_modes)

        ckpt_tag = self._resolve_stage2_gallery_ckpt_tag(ckpt_path)
        if ckpt_path is not None:
            gallery_bank.meta["ckpt_path"] = ckpt_path
        if ckpt_tag is not None:
            gallery_bank.meta["ckpt_tag"] = ckpt_tag

        return {
            "gallery_bank": gallery_bank,
            "gallery_save_dir": gallery_save_dir,
            "ckpt_path": ckpt_path,
        }


    def eval_gallery_bank(
            self,
            layout_cfg,
            feature_cfg=None,
            retrieval_eval_cfg=None,
            gallery_save_dir=None,
            load_if_exists=True,
            save_gallery=True,
            init_datasets=True,
            load_ckpt=False,
            gallery_root_dir=None,
            gallery_name_prefix=None,
    ):
        """
        Stage 2版 gallery build/load + retrieval eval orchestration。
        """
        if retrieval_eval_cfg is not None and not isinstance(retrieval_eval_cfg, Stage2RetrievalEvalConfig):
            retrieval_eval_cfg = Stage2RetrievalEvalConfig(**retrieval_eval_cfg)

        if feature_cfg is None:
            feature_cfg = Stage2ReferenceGalleryFeatureConfig()
        elif not isinstance(feature_cfg, Stage2ReferenceGalleryFeatureConfig):
            feature_cfg = Stage2ReferenceGalleryFeatureConfig(**feature_cfg)

        if retrieval_eval_cfg is not None:
            feature_cfg.build_faiss = True

        gallery_state = self.build_or_load_gallery_bank(
            layout_cfg=layout_cfg,
            feature_cfg=feature_cfg,
            gallery_save_dir=gallery_save_dir,
            load_if_exists=load_if_exists,
            save_gallery=save_gallery,
            init_datasets=init_datasets,
            load_ckpt=load_ckpt,
            gallery_root_dir=gallery_root_dir,
            gallery_name_prefix=gallery_name_prefix,
        )
        gallery_bank = gallery_state["gallery_bank"]

        eval_res = None
        if retrieval_eval_cfg is not None:
            retrieval_evaluator = Stage2RetrievalEvaluator(
                trainer=self,
                gallery_bank=gallery_bank,
                logger=self.logger,
            )
            eval_res = retrieval_evaluator.evaluate(eval_cfg=retrieval_eval_cfg)

        gallery_state["eval_res"] = eval_res
        return gallery_state


    def train(self):
        """Stage 2训练主循环"""
        opt = self.opt

        print("\n" + "🚀"*40)
        print("开始 Stage 2 训练: Grid HashFit")
        print("🚀"*40 + "\n")

        # 0. 初始化GradScaler（如果使用autocast）
        if opt.autocast:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 启用混合精度训练 (AMP)")

        # 1. 优化器
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam')

        # 2. 加载checkpoint（如果继续训练）
        train_ckpt_modules = self._make_train_checkpoint_modules()
        begin_epoch = self._load_checkpoint(
            opt.load2train,
            train_ckpt_modules,
            self.optimizer,
            mode='train'
        )

        # 3. 初始化日志
        self._init_logger()

        # 4. 初始化数据集
        self._init_datasets(create_train_loader=False)

        # 创建DataLoader
        self.sat_dataloader = torch.utils.data.DataLoader(
            self.sat_dataset,
            batch_size=opt.batchsize_sat,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(opt.num_worker > 0),
        )

        self.uav_dataloader_train = torch.utils.data.DataLoader(
            self.uav_dataset_train,
            batch_size=opt.batchsize_uav,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(opt.num_worker > 0),
        )

        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=opt.batchsize_uav,
            num_workers=opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(opt.num_worker > 0)
        )

        # 4.5 初始化4d坐标归一化器
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        from trainer_depends.utils.util_udf_computer_euc5d import UDFComputer
        self.udf_compter_5d = UDFComputer(norm_processor=self.coord_normer)
        # 5. 配置Loss
        loss_mse = torch.nn.MSELoss(reduction='mean')

        # 6. 训练循环
        num_epochs = opt.num_epochs
        since = time.time()
        save_freq = max(1, int(getattr(self.opt, "save_freq", 10)))
        step = 0

        self.logger.info(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(begin_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            for it, batch in tqdm.tqdm(enumerate(self.sat_dataloader)):
                # 获取sat数据
                satimgs = batch[0].to(self.device)
                coords_sat = batch[1].to(self.device)  # [B, 4]

                # 获取uav数据
                batch_uav = next(iter(self.uav_dataloader_train))
                uavimgs = batch_uav[0].to(self.device)
                coords_uav = batch_uav[1].to(self.device)  # [B, 4]

                # 合并坐标并转到欧式空间
                coords_all = torch.cat([coords_sat, coords_uav], dim=0)  # [2B, 4]
                # coords_all_5d = self.coord_normer.raw_to_norm(coords_all)
                coords_all_6d = self.coord_normer.raw_to_norm(coords_all,append_linear_rot=True)
                # coords_all_rot_rad = coords_all[:,2:3]

                # 从Grid提取特征
                feats_grid = self._get_feats_fm_grid(torch.concatenate([coords_all_6d[:,:2],coords_all_6d[:,-1:]],dim=-1))  # [2B, feat_dim]
                # 位置编码
                # version1:
                coords_all_encoded = self.pos_encoder_grid(
                    coords_all_6d[:,:5],
                )  # [2B, coord_encoded_dim]
                # Grid MLP调制
                feats_grid = self.grid_mlp(
                    inputs=feats_grid,
                    condition_features=coords_all_encoded
                )  # [2B, feat_dim]

                # 提取视觉特征（冻结）
                feats_vis = self._get_feats_fm_imgs(
                    torch.cat([satimgs, uavimgs], dim=0)
                )  # [2B, feat_dim]

                # L2归一化
                feats_grid = TF.normalize(feats_grid, dim=-1)

                # 计算loss（Grid特征拟合视觉特征）
                loss = loss_mse(feats_grid.squeeze(), feats_vis.squeeze()) * 1000

                # 反向传播
                self.optimizer.zero_grad()
                if opt.autocast:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 记录loss
                if it % 10 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss.item(), step)
                step += 1

            # 每个epoch结束后
            if (epoch % 1 == 0) and (epoch > 0):
                # 统计信息
                grid_feats_grad = self.grid.codebook.feats.grad
                self.logger.info(f"Grid feats grad L2 norm: {torch.linalg.norm(grid_feats_grad).item():.6f}")
                self.logger.info(f"Grid feats value.max(): {feats_grid.max().item():.4f}")

                mean_dino = torch.mean(feats_vis, dim=0)
                mean_grid = torch.mean(feats_grid, dim=0)
                l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                cosine_sim = TF.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                self.logger.info(f"mean值- L2距离: {l2_distance:.4f}")
                self.logger.info(f"mean值- 余弦相似度: {cosine_sim:.4f}")

                var_per_dim_grid = torch.var(feats_grid, dim=0)
                var_per_dim_dino = torch.var(feats_vis, dim=0)
                self.logger.info(f"HashGrid方差均值: {var_per_dim_grid.mean().item():.4e}")
                self.logger.info(f"Vis方差均值: {var_per_dim_dino.mean().item():.4e}")

                # 训练中快速做一次RC定位精度检查。
                # 这里走的是临时建库 + 检索评估路径，不会保存gallery bank。
                if (epoch % 10 == 0):
                    rc_eval_res = self.test_rc_localization(overlap=0.5, fixed_scale=None)
                    self.logger.info(
                        f"[Train Eval][RC] N={int(rc_eval_res['n_queries'])} "
                        + " | ".join(
                            [f"R@{int(k)}={float(v) * 100.0:.3f}%" for k, v in rc_eval_res['recall@k'].items()]
                        )
                    )
                    self.logger.info(
                        f"[Train Eval][RC] error_rc_norm={rc_eval_res['error_rc_norm']:.6f}, "
                        f"error_rc_meter={rc_eval_res['error_rc_meter']:.3f}m"
                    )

                # 保存
                if (epoch % save_freq == 0):
                    self._save_checkpoint(
                        epoch,
                        train_ckpt_modules,
                        self.optimizer
                    )

            # 日志
            self.logger.info(f'loss={loss.item():.6f}')
            time_elapsed = time.time() - since
            since = time.time()
            self.logger.info(f'epoch {epoch} 完成，耗时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
            self.logger.info('-' * 50)

            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss.item(), epoch)

        self.logger.info("✅ Stage 2 训练完成！")



    def test(self):
        """
        测试并可视化 INGP 的拟合能力
        """
        import yaml
        print("\n" + "🧪"*40)
        print("开始 Stage 2 测试: INGP 拟合能力可视化")
        print("🧪"*40 + "\n")

        # 1. 初始化数据集和坐标归一化器 (test模式下也需要)
        self._init_datasets(create_train_loader=False)
        from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
        self.coord_normer = CoordsNormProcessor(self.sat_dataset)
        self.uav_dataloader_test = torch.utils.data.DataLoader(
            self.uav_dataset_test,
            batch_size=self.opt.batchsize_uav,
            num_workers=self.opt.num_worker,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(self.opt.num_worker > 0)
        )

        # 2. 加载checkpoint
        self._load_checkpoints_for_test()

        # 3. 开始测试可视化
        # self.visualize_similarity_in_3d(delta=0.1)

        # 4. 开始测试检索定位
        # 4.1 测试RC定位（消除旋转差异）
        # print("\n" + "🔹"*40)
        # print("测试1: RC定位（消除旋转差异）")
        # print("🔹"*40)
        # self.test_rc_localization(overlap=0.5, fixed_scale=None)

        # 4.2 测试RC + 旋转定位
        # print("\n" + "🔹"*40)
        # print("测试2: RC + 旋转定位")
        # print("🔹"*40)
        # self.test_rc_rot_localization(overlap=0.5, delta_rot_deg=10, fixed_scale=None)

        # 4.3 测试RC + 尺度定位
        # print("\n" + "🔹"*40)
        # print("测试3: RC + 尺度定位")
        # print("🔹"*40)
        # self.test_rc_scale_localization(overlap=0.5, n_scales=3)

        # 4.4 测试RC + 旋转 + 尺度定位
        print("\n" + "🔹"*40)
        print("测试4: RC + 旋转 + 尺度定位")
        print("🔹"*40)
        self.test_rc_rot_scale_localization(overlap=0.5, delta_rot_deg=10, n_scales=4)


    def _load_checkpoints_for_test(self):
        """
        测试时加载checkpoint的统一方法

        加载逻辑：
        1. Stage 2 checkpoint：仅恢复 grid, grid_mlp
        2. Stage 1 checkpoint：恢复 vis_encoder, vis_aggregator
        """
        print("\n" + "="*80)
        print("加载测试用的checkpoint")
        print("="*80)

        stage2_ckpt_path = self._get_stage2_checkpoint_path()

        if stage2_ckpt_path:
            print(f"\n📦 Stage 2 checkpoint: {stage2_ckpt_path}")
            self._load_checkpoint(
                {
                    'grid': stage2_ckpt_path,
                    'grid_mlp': stage2_ckpt_path,
                },
                self.param2optimize,
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 2的checkpoint，无法进行测试。")

        stage1_ckpt_path = self._get_stage1_checkpoint_path(stage2_ckpt_path)
        if stage1_ckpt_path:
            print(f"\n📦 Stage 1 checkpoint: {stage1_ckpt_path}")
            self._load_checkpoint(
                {
                    'vis_encoder': stage1_ckpt_path,
                    'vis_aggregator': stage1_ckpt_path,
                },
                self.param2freeze,
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 1的checkpoint，无法进行测试。")

        print("\n" + "="*80)
        print("✅ 测试checkpoint加载完成")
        print("="*80 + "\n")


    def _get_stage2_checkpoint_path(self):
        """获取Stage 2的checkpoint路径"""
        # 优先使用命令行参数指定的路径
        if hasattr(self.opt, 'load2test') and self.opt.load2test:
            print(f"从opt.load2test读取: {self.opt.load2test}")
            return self.opt.load2test

        # 否则从实验目录中找最新的checkpoint
        if self.exp_dir2save and os.path.exists(self.exp_dir2save):
            ckpts = [f for f in os.listdir(self.exp_dir2save) if f.startswith('epoch')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace('epoch','').split('.')[0]))
                ckpt_path = os.path.join(self.exp_dir2save, ckpts[-1])
                print(f"从实验目录读取: {ckpt_path}")
                return ckpt_path

        print(f"⚠️  未找到Stage 2 checkpoint:")
        print(f"   opt.load2test = {getattr(self.opt, 'load2test', 'NOT SET')}")
        print(f"   exp_dir2save = {self.exp_dir2save}")
        return None


    def _get_stage1_checkpoint_path(self, stage2_ckpt_path):
        """
        获取测试时使用的 Stage 1 checkpoint 路径。

        优先级：
        1. 当前运行配置中的 opt.load_stage1_ckpt
        2. Stage 2 checkpoint 同目录下 opts.yaml 中记录的 load_stage1_ckpt
        """
        import yaml

        if getattr(self.opt, 'load_stage1_ckpt', None):
            print(f"从opt.load_stage1_ckpt读取: {self.opt.load_stage1_ckpt}")
            return self.opt.load_stage1_ckpt

        if stage2_ckpt_path:
            stage2_exp_dir = os.path.dirname(stage2_ckpt_path)
            stage2_opts_path = os.path.join(stage2_exp_dir, 'opts.yaml')

            if os.path.exists(stage2_opts_path):
                try:
                    with open(stage2_opts_path, 'r', encoding='utf-8') as f:
                        stage2_opts = yaml.safe_load(f)

                    if 'exp_setting' in stage2_opts:
                        stage1_path = stage2_opts['exp_setting'].get('load_stage1_ckpt')
                        if stage1_path:
                            print(f"从opts.yaml读取Stage 1路径: {stage2_opts_path}")
                            return stage1_path
                except Exception as e:
                    print(f"⚠️  读取opts.yaml失败: {e}")

        return None


    def visualize_similarity_in_3d(self, metric='euclidean',delta=0.1):
        """#todo:将这个函数移除为外部工具函数
        在3D空间(r, c, rot)中可视化相似度分布

        Args:
            metric: 可视化的度量指标
                - 'cosine': 余弦相似度 (越大越好，范围[-1, 1])
                - 'euclidean': 欧式距离 (越小越好，范围[0, +∞))
        """
        # 0. 确保所有模型处于评估模式
        for model in self.param2optimize.values():
            model.eval()
        for model in self.param2freeze.values():
            model.eval()

        print(f"\n{'='*80}")
        print(f"可视化度量指标: {metric.upper()}")
        print(f"{'='*80}\n")

        # 1. 从测试集中抽取一个UAV样本
        uav_img, uav_coords_4d = next(iter(self.uav_dataloader_test))
        uav_img = uav_img[0].to(self.device).unsqueeze(0)
        uav_coords_4d = uav_coords_4d[0].to(self.device)

        # 2. 提取"真值"视觉特征
        feats_vis_gt = self._get_feats_fm_imgs(uav_img)
        feats_vis_gt = TF.normalize(feats_vis_gt, dim=-1)

        # 3. 创建评估网格
        n_pts_grid = 32
        nr_center, nc_center, rot_center, scale_val = uav_coords_4d

        nr_range = torch.linspace(nr_center - delta, nr_center + delta, n_pts_grid, device=self.device)
        nc_range = torch.linspace(nc_center - delta, nc_center + delta, n_pts_grid, device=self.device)
        rot_range = torch.linspace(rot_center - torch.pi/2, rot_center + torch.pi/2, n_pts_grid, device=self.device)

        grid_nr, grid_nc, grid_rot = torch.meshgrid(nr_range, nc_range, rot_range, indexing='ij')

        grid_coords_4d = torch.stack([
            grid_nr.flatten(), grid_nc.flatten(), grid_rot.flatten(),
            torch.full_like(grid_nr.flatten(), scale_val)
        ], dim=-1)

        # 4. 遍历网格，计算INGP特征和度量值
        with torch.no_grad():
            # 转换到6D空间（包含线性旋转）
            coords_all_6d = self.coord_normer.raw_to_norm(grid_coords_4d, append_linear_rot=True)

            # Grid输入：前2维(nr, nc) + 最后1维(log_scale)
            grid_coords_3d = torch.cat([coords_all_6d[:, 0:2], coords_all_6d[:, -1:]], dim=-1)

            # 从Grid提取特征
            feats_grid = self._get_feats_fm_grid(grid_coords_3d)

            # 位置编码：使用前5维（nr, nc, cos, sin, log_scale）
            coords_all_encoded = self.pos_encoder_grid(coords_all_6d[:, :5])

            # Grid MLP调制
            feats_grid = self.grid_mlp(inputs=feats_grid, condition_features=coords_all_encoded)
            feats_grid = TF.normalize(feats_grid, dim=-1)

            # 计算两种度量
            similarities = TF.cosine_similarity(feats_grid, feats_vis_gt.expand_as(feats_grid), dim=-1)
            dists = torch.norm(feats_grid - feats_vis_gt.expand_as(feats_grid), p=2, dim=-1)

        # 5. 根据度量类型选择可视化的值
        if metric == 'cosine':
            metric_values = similarities
            best_idx = torch.argmax(metric_values)
            best_value = metric_values[best_idx]
            metric_name = 'Cosine Similarity'
            colorscale = 'Viridis'  # 黄色表示高相似度
        elif metric == 'euclidean':
            metric_values = dists
            best_idx = torch.argmin(metric_values)
            best_value = metric_values[best_idx]
            metric_name = 'Euclidean Distance'
            colorscale = 'Viridis_r'  # 反转颜色，使蓝色表示小距离
        else:
            raise ValueError(f"不支持的度量类型: {metric}，请使用 'cosine' 或 'euclidean'")

        coord_pred_best = grid_coords_4d[best_idx]

        # 8. 可视化
        try:
            import plotly.graph_objects as go

            colors = metric_values.cpu().numpy()

            # Trace 1: 度量值点云 (背景)
            fig = go.Figure(data=go.Scatter3d(
                x=grid_coords_4d[:, 0].cpu().numpy(),
                y=grid_coords_4d[:, 1].cpu().numpy(),
                z=grid_coords_4d[:, 2].cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale=colorscale,
                    colorbar=dict(title=metric_name),
                    opacity=0.6
                ),
                name=f'{metric_name} Cloud'
            ))

            # Trace 2: GT 真值 (红色菱形)
            fig.add_trace(go.Scatter3d(
                x=[nr_center.cpu().item()],
                y=[nc_center.cpu().item()],
                z=[rot_center.cpu().item()],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name='Ground Truth',
                text=['GT'],
                textposition="top left"
            ))

            # Trace 3: 预测最佳值 (青色叉号)
            fig.add_trace(go.Scatter3d(
                x=[coord_pred_best[0].cpu().item()],
                y=[coord_pred_best[1].cpu().item()],
                z=[coord_pred_best[2].cpu().item()],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='cyan',
                    symbol='cross',
                    line=dict(width=2, color='blue')
                ),
                name=f'Best {metric_name}',
                text=[f'{best_value:.3f}'],
                textposition="top right"
            ))

            # 更新布局
            fig.update_layout(
                title=f'{metric_name} Distribution (Best: {best_value:.4f})',
                scene=dict(
                    xaxis_title='Normalized Row',
                    yaxis_title='Normalized Col',
                    zaxis_title='Rotation (rad)'
                )
            )

            fig.show()

        except ImportError:
            print("请安装 plotly 以进行3D可视化: pip install plotly")

        # 6. 打印分析结果
        print(f"\n=== Peak Analysis ({metric_name}) ===")
        print(f"GT Center : r={nr_center:.4f}, c={nc_center:.4f}, rot={rot_center:.4f}")
        print(f"Predicted Best : r={coord_pred_best[0]:.4f}, c={coord_pred_best[1]:.4f}, rot={coord_pred_best[2]:.4f}")
        print(f"Best {metric_name}: {best_value:.4f}")

        # 额外打印两种度量的对比
        max_sim_val = similarities[best_idx]
        min_dist_val = dists[best_idx]
        print(f"\n对应位置的两种度量:")
        print(f"  Cosine Similarity: {max_sim_val:.4f}")
        print(f"  Euclidean Distance: {min_dist_val:.4f}")


    def _make_stage2_gallery_feature_cfg(self):
        return Stage2ReferenceGalleryFeatureConfig(
            chunk_size_coords=512,
            normalize_feats=True,
            build_faiss=True,
            show_progress=True,
        )


    def _make_stage2_retrieval_eval_cfg(self, **overrides):
        cfg = {
            'use_train_uav': False,
            'batch_size': int(getattr(self.opt, 'batchsize_uav', 32)),
            'num_workers': int(getattr(self.opt, 'num_worker_eval', 0)),
            'query_rot2uniform': False,
            'query_scale2uniform': False,
            'k_values': (1, 5, 10, 20, 50, 256, 512, 1024),
            'dist_th': None,
            'max_queries': None,
            'print_results': False,
            'report_title': 'Stage2 Retrieval Eval',
            'report_rc_meter': True,
            'report_rot_error': False,
            'report_scale_error': False,
        }
        cfg.update(overrides)
        return Stage2RetrievalEvalConfig(**cfg)


    def test_rc_localization(self, overlap=0.5, fixed_scale=None):
        """测试RC定位精度（通过逆向旋转UAV图像消除旋转差异）"""
        print("\n" + "="*80)
        print("测试 RC 定位精度 (通过逆向旋转消除rot差异)")
        print("="*80)
        print("策略: 特征库 rot=0, UAV图像逆向旋转到 rot=0")
        print(f"重叠度: {overlap}, 固定scale: {fixed_scale}")

        layout_cfg = Stage2ReferenceGalleryLayoutConfig(
            mode='rc',
            overlap=overlap,
            fixed_rot=0.0,
            fixed_scale=fixed_scale,
        )
        eval_cfg = self._make_stage2_retrieval_eval_cfg(
            query_rot2uniform=True,
            report_title='Stage2 RC Localization',
        )
        eval_res = self.evaluate_stage2_gallery_bank(
            layout_cfg=layout_cfg,
            feature_cfg=self._make_stage2_gallery_feature_cfg(),
            eval_cfg=eval_cfg,
        )
        return {
            'recall@k': eval_res['recall@k'],
            'error_rc_norm': eval_res['error_rc_norm'],
            'error_rc_meter': eval_res['error_rc_meter'],
            'n_queries': eval_res['n_queries'],
        }


    def test_rc_rot_localization(self, overlap=0.5, delta_rot_deg=10, fixed_scale=None):
        """测试RC和旋转定位精度（支持旋转维度）"""
        print("\n" + "="*80)
        print("测试 RC + 旋转 定位精度")
        print("="*80)
        print("策略: 特征库包含多个旋转角度，直接使用原始UAV图像")
        print(f"重叠度: {overlap}, 旋转间隔: {delta_rot_deg}°, 固定scale: {fixed_scale}")

        layout_cfg = Stage2ReferenceGalleryLayoutConfig(
            mode='rc_rot',
            overlap=overlap,
            fixed_scale=fixed_scale,
            delta_rot_deg=delta_rot_deg,
        )
        eval_cfg = self._make_stage2_retrieval_eval_cfg(
            query_rot2uniform=False,
            report_title='Stage2 RC + Rot Localization',
            report_rot_error=True,
        )
        eval_res = self.evaluate_stage2_gallery_bank(
            layout_cfg=layout_cfg,
            feature_cfg=self._make_stage2_gallery_feature_cfg(),
            eval_cfg=eval_cfg,
        )
        return {
            'recall@k': eval_res['recall@k'],
            'error_rc_norm': eval_res['error_rc_norm'],
            'error_rc_meter': eval_res['error_rc_meter'],
            'error_rot_deg': eval_res['error_rot_deg'],
        }


    def test_rc_scale_localization(self, overlap=0.5, n_scales=3):
        """测试RC和尺度定位精度"""
        print("\n" + "="*80)
        print("测试 RC + 尺度 定位精度")
        print("="*80)
        print("策略: 特征库包含多个尺度，UAV图像逆向旋转到 rot=0")
        print(f"重叠度: {overlap}, 尺度数: {n_scales}")

        layout_cfg = Stage2ReferenceGalleryLayoutConfig(
            mode='rc_scale',
            overlap=overlap,
            fixed_rot=0.0,
            n_scales=n_scales,
        )
        eval_cfg = self._make_stage2_retrieval_eval_cfg(
            query_rot2uniform=True,
            report_title='Stage2 RC + Scale Localization',
            report_scale_error=True,
        )
        eval_res = self.evaluate_stage2_gallery_bank(
            layout_cfg=layout_cfg,
            feature_cfg=self._make_stage2_gallery_feature_cfg(),
            eval_cfg=eval_cfg,
        )
        return {
            'recall@k': eval_res['recall@k'],
            'error_scale_normed': eval_res['error_scale_normed'],
        }


    def test_rc_rot_scale_localization(self, overlap=0.5, delta_rot_deg=10, n_scales=3):
        """测试RC、旋转和尺度定位精度"""
        print("\n" + "="*80)
        print("测试 RC + 旋转 + 尺度 定位精度")
        print("="*80)
        print("策略: 特征库包含多尺度和多旋转角度，直接使用原始UAV图像")
        print(f"重叠度: {overlap}, 旋转间隔: {delta_rot_deg}°, 尺度数: {n_scales}")

        layout_cfg = Stage2ReferenceGalleryLayoutConfig(
            mode='rc_rot_scale',
            overlap=overlap,
            delta_rot_deg=delta_rot_deg,
            n_scales=n_scales,
        )
        eval_cfg = self._make_stage2_retrieval_eval_cfg(
            query_rot2uniform=False,
            report_title='Stage2 RC + Rot + Scale Localization',
            report_rot_error=True,
            report_scale_error=True,
        )
        eval_res = self.evaluate_stage2_gallery_bank(
            layout_cfg=layout_cfg,
            feature_cfg=self._make_stage2_gallery_feature_cfg(),
            eval_cfg=eval_cfg,
        )
        return {
            'recall@k_rc': eval_res['recall@k'],
            'error_rc_norm': eval_res['error_rc_norm'],
            'error_rot_deg': eval_res['error_rot_deg'],
            'error_scale_normed': eval_res['error_scale_normed'],
        }


if __name__ == "__main__":
    import argparse

    # 先解析 stage2 脚本自己的控制参数，再把剩余参数交给通用 YAML parser。
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_only', action='store_true', help='是否只运行测试模式')
    parser.add_argument(
        '--test_mode',
        type=str,
        default='legacy_test',
        choices=('legacy_test', 'gallery_bank'),
        help='测试模式：legacy_test 为旧 test() 入口，gallery_bank 为显式建库+评测入口',
    )
    args, remaining_argv = parser.parse_known_args()

    # 如果没有显式指定配置文件，默认使用 stage2 配置，而不是 parser 的 stage1 默认值。
    if '--p_yaml' not in remaining_argv:
        # remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/trainer_depends/configs/stage2_INGP.yaml'])
        remaining_argv.extend(['--p_yaml', '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage2_INGP_visloc.yaml'])

    # 将剩余参数交给通用配置解析器，避免 GridHashFitTrainer 再次读取错误默认值。
    sys.argv = [sys.argv[0]] + remaining_argv

    #todo:modify manually
    args.test_only = False
    args.test_mode = 'gallery_bank'
    exp_name_override = None

    opt = get_parse(print_summary=False)
    if exp_name_override:
        opt.exp_name = exp_name_override

    trainer = GridHashFitTrainer(opt=opt)
    if not args.test_only:
        trainer.train()
    elif args.test_mode == 'legacy_test':
        trainer.test()
    elif args.test_mode == 'gallery_bank':
        gallery_layout_cfg = Stage2ReferenceGalleryLayoutConfig(
            mode='rc_rot_scale',
            overlap=0.5,
            delta_rot_deg=5,
            n_scales=4,
        )
        gallery_feature_cfg = trainer._make_stage2_gallery_feature_cfg()
        gallery_feature_cfg.build_faiss = True
        gallery_feature_cfg.show_progress = True

        retrieval_eval_cfg = trainer._make_stage2_retrieval_eval_cfg(
            use_train_uav=False,
            query_rot2uniform=False,
            query_scale2uniform=False,
            report_title='Stage2 Gallery Eval',
            report_rot_error=True,
            report_scale_error=True,
        )

        gallery_state = trainer.eval_gallery_bank(
            layout_cfg=gallery_layout_cfg,
            feature_cfg=gallery_feature_cfg,
            retrieval_eval_cfg=retrieval_eval_cfg,
            gallery_save_dir=None,
            load_if_exists=True,
            save_gallery=True,
            init_datasets=True,
            load_ckpt=True,
            gallery_name_prefix=None,
        )
        print(f"[Stage2 Gallery Demo] save_dir={gallery_state['gallery_save_dir']}")
    else:
        raise ValueError(f"Unknown test_mode: {args.test_mode}")
