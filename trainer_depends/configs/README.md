# 配置文件说明

## 📁 配置文件列表

### 训练配置
- `stage1_visual_encoder.yaml` - Stage 1: 视觉编码器训练配置
- `stage2_grid_hashfit.yaml` - Stage 2: Grid HashFit训练配置
- `stage3_metric_net.yaml` - Stage 3: MetricNet训练配置

### Grid配置
- `nerf_hash.yaml` - INGP (Instant Neural Graphics Primitives) Grid配置

## 🚀 使用方法

### Stage 1: 训练视觉特征聚合器

```bash
cd /home/data/zwk/pyproj_neuloc_v0

# 使用默认配置
python trainers/stage1_visual_encoder.py

# 或指定配置文件
python trainers/stage1_visual_encoder.py \
    --p_yaml trainer_depends/configs/stage1_visual_encoder.yaml
```

### Stage 2: 训练Grid和Grid MLP

```bash
# 需要先训练Stage 1并获得checkpoint
python trainers/stage2_grid_hashfit.py \
    --p_yaml trainer_depends/configs/stage2_grid_hashfit.yaml \
    --load_stage1_ckpt .exps/stage1_visual_encoder/checkpoint_epoch99.pth
```

### Stage 3: 训练MetricNet

```bash
# 需要先训练Stage 1和Stage 2
python trainers/stage3_metric_net.py \
    --p_yaml trainer_depends/configs/stage3_metric_net.yaml \
    --load_stage1_ckpt .exps/stage1_visual_encoder/checkpoint_epoch99.pth \
    --load_stage2_ckpt .exps/stage2_grid_hashfit/checkpoint_epoch99.pth
```

## ⚙️ 配置文件结构

### exp_setting - 实验配置
```yaml
exp_setting:
  exps_dir: .exps              # 实验保存目录（相对于项目根目录）
  exp_name: stage1_xxx         # 实验名称
  save_freq: 5                 # 每隔多少个epoch保存checkpoint
  tensorboard: True            # 是否启用tensorboard
  val: False                   # 是否进行验证（暂未实现）
  val_freq: 1                  # 验证频率

  # Checkpoint加载
  load_stage1_ckpt: ""         # Stage 1的checkpoint路径（Stage 2/3使用）
  load_stage2_ckpt: ""         # Stage 2的checkpoint路径（Stage 3使用）
  load2test: ""                # 测试时加载的checkpoint
  load2train: ""               # 继续训练时加载的checkpoint
```

### data_setting - 数据配置
```yaml
data_setting:
  imgsize2net: 224             # 输入网络的图像大小
  satimgsize2crop: 244         # 卫星图crop大小
  n_rand2sample_per_pos: 256   # 每个正样本采样的负样本数
```

### scenes_setting - 场景配置
```yaml
scenes_setting:
  sampling_strategy: "random"  # 采样策略: round_robin, random, weighted

  scenes:
    - name: "zurich"
      p_satinfo_json: /path/to/sat.json
      p_uavinfo_json: /path/to/uav.json
      p_uav_geocsv: /path/to/geo.csv
      weight: 1.0                # 采样权重（weighted策略使用）

    - name: "zuchwil"
      p_satinfo_json: /path/to/sat.json
      p_uavinfo_json: /path/to/uav.json
      p_uav_geocsv: /path/to/geo.csv
      weight: 1.0
```

### hardware_setting - 硬件配置
```yaml
hardware_setting:
  autocast: False              # 是否使用混合精度训练
  batchsize_sat: 256           # 卫星图batch size
  batchsize_uav: 256           # UAV图batch size
  gpu_ids: '0'                 # GPU IDs
  num_worker: 8                # DataLoader worker数量
```

### learning_setting - 学习配置
```yaml
learning_setting:
  num_epochs: 100              # 训练轮数
```

### network_setting - 网络配置
```yaml
network_setting:
  backbone: dinov3             # Backbone类型: dinov2, dinov3, ViTB-224等
  aggregator_type: salad       # 聚合器类型: salad, g2m
  freeze_grid: True            # 是否冻结Grid（Stage 3使用）
```

## 📝 配置优先级

命令行参数 > YAML文件参数 > 默认参数

例如：
```bash
python trainers/stage1_visual_encoder.py \
    --p_yaml trainer_depends/configs/stage1_visual_encoder.yaml \
    --exp_name my_experiment \
    --num_epochs 200 \
    --batchsize_sat 128
```

## 🔧 自定义配置

### 方法1: 修改YAML文件
直接编辑 `trainer_depends/configs/stage1_visual_encoder.yaml`

### 方法2: 创建新的YAML文件
```bash
cp trainer_depends/configs/stage1_visual_encoder.yaml my_config.yaml
# 编辑 my_config.yaml
python trainers/stage1_visual_encoder.py --p_yaml my_config.yaml
```

### 方法3: 命令行覆盖
```bash
python trainers/stage1_visual_encoder.py \
    --p_yaml trainer_depends/configs/stage1_visual_encoder.yaml \
    --exp_name custom_exp \
    --num_epochs 200
```

## 📊 实验输出

所有实验结果保存在 `.exps/` 目录下：

```
.exps/
├── stage1_visual_encoder/
│   ├── checkpoint_epoch99.pth
│   ├── train.log
│   └── train_tensorboard.log/
├── stage2_grid_hashfit/
│   └── ...
└── stage3_metric_net/
    └── ...
```

## ⚠️ 重要说明

1. **相对路径**: `exps_dir` 使用相对路径 `.exps`，基于项目根目录
2. **数据路径**: `scenes_setting` 中的数据路径需要根据实际情况修改
3. **Checkpoint路径**: Stage 2/3需要正确指定前序stage的checkpoint路径
4. **独立性**: 这些配置文件独立于 `train_img_encoder/` 中的旧配置文件

---

**位置**: `/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/`
**更新日期**: 2025-12-03
