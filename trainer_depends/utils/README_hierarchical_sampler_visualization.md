# 分层坐标采样器可视化验证工具

## 概述

此工具用于验证和可视化分层坐标采样器（HierarchicalCoordSampler）的行为，使用真实的数据集配置。

## 文件说明

### 核心文件
- `util_hierarchical_coord_sampler.py` - 分层坐标采样器实现
- `util_visualize_hierarchical_sampler_with_real_data.py` - 可视化验证脚本

## 使用方法

### 方式1: 单个样本详细可视化（默认）

```bash
python trainer_depends/utils/util_visualize_hierarchical_sampler_with_real_data.py
```

**输出:**
- `hierarchical_sampling_real_data.png` - 2D投影图（4个子图）
- `hierarchical_sampling_real_data.html` - 3D交互式图表（可在浏览器中打开）
- 控制台输出各层的统计信息和标准差限制警告

### 方式2: 多样本批量测试

```bash
python trainer_depends/utils/util_visualize_hierarchical_sampler_with_real_data.py --mode multiple --n_samples 10
```

这将测试10个真实样本的采样效果，并显示每个样本的采样统计。

## 功能特性

### 1. 数据集初始化
- 与 `BaseTrainer._init_datasets()` 完全一致
- 从 `trainer_depends/configs/stage3_metric_net.yaml` 读取配置
- 加载真实的卫星数据集和UAV数据集

### 2. 采样器配置
- 使用与训练时完全相同的参数：
  - `base_rc_std`: RC维度基准标准差（默认0.01）
  - `base_rot_std_rad`: 旋转维度基准标准差（默认π/4）
  - `base_log_scale_std`: 对数尺度基准标准差（默认0.2）
  - `num_uniform_samples`: 全局均匀采样数量（默认128）

### 3. 可视化功能

#### verbose模式
启用后会显示每一层的：
- 配置的标准差
- 最大允许的标准差
- 实际有效的标准差
- 标准差被限制的百分比（如果发生）

#### 3D交互式可视化
- 使用plotly生成3D散点图
- 显示采样点在(row, col, rotation)空间中的分布
- 可以旋转、缩放、查看具体数值

#### 2D投影可视化
- 4个子图显示不同维度的投影：
  - (Row, Col) 投影
  - (Row, Rotation) 投影
  - (Col, Rotation) 投影
  - (Row, Scale) 投影

### 4. 统计分析
对每一层计算：
- RC距离均值和标准差
- 旋转距离均值和标准差
- 尺度比例均值和标准差

## 预期发现

运行此工具后，你可能会发现：

### 标准差被限制的问题
由于`safety_factor=3.0`的限制，对于归一化坐标范围`[-0.5, 0.5]`：

```
层级 'slope':
  配置rc_std: 0.2000
  实际有效rc_std: 0.1333
  ⚠️  警告: 实际标准差被限制了 33.5%

层级 'rim':
  配置rc_std: 0.4000
  实际有效rc_std: 0.1333
  ⚠️  警告: 实际标准差被限制了 66.7%
```

**结果:** slope和rim的RC采样范围几乎相同！

### 解决方案

#### 方案1: 调整safety_factor
```python
sampler = HierarchicalCoordSampler(
    ...
    safety_factor=6.0,  # 从3.0增加到6.0
)
```

#### 方案2: 调整multiplier比例
修改 `util_hierarchical_coord_sampler.py` 中的默认配置：
```python
strategy_definition = [
    {'name': 'bottom', 'rc_multiplier': 1.0, ...},
    {'name': 'slope', 'rc_multiplier': 3.0, ...},   # 从10.0改为3.0
    {'name': 'rim', 'rc_multiplier': 5.0, ...},     # 从20.0改为5.0
]
```

#### 方案3: 依赖旋转和尺度维度
保持RC采样保守，主要通过旋转和尺度维度来区分不同层级（因为旋转是周期性的，没有边界限制）。

## 与训练流程的一致性

此验证脚本与`trainers/stage3_metric_net.py:789-798`中的采样器初始化完全一致：

```python
# 训练代码（stage3_metric_net.py）
self.coord_sampler = create_hierarchical_sampler_from_dataset(
    sat_dataset=self.sat_dataset,
    base_rc_std=getattr(opt, 'sampler_base_rc_std', 0.01),
    base_rot_std_rad=getattr(opt, 'sampler_base_rot_std_rad', 3.14159/4),
    base_log_scale_std=getattr(opt, 'sampler_base_log_scale_std', 0.2),
    num_uniform_samples=getattr(opt, 'sampler_num_uniform',128),
    device=self.device
)
```

## 注意事项

1. **坐标系统**: 使用归一化坐标系统（与数据集一致）
2. **设备**: 默认使用'cpu'进行可视化（避免GPU占用）
3. **数据集**: 使用测试集的GT坐标进行验证
4. **配置文件**: 确保`trainer_depends/configs/stage3_metric_net.yaml`存在且路径正确