# 地图网格划分和UAV位点可视化工具

## 功能说明

这个可视化脚本 (`visualize_grid_and_uav_points.py`) 可以：

1. **绘制2D网格划分**：基于 `SubspaceSampler` 的 `n_coarse` 参数在地图上绘制网格
2. **显示UAV位点**：
   - 训练集位点（蓝色圆点）
   - 测试集位点（红色三角）
3. **生成热力图**（可选）：显示UAV位点的密度分布
4. **保存高质量图像**：支持自定义分辨率和尺寸

## 快速使用

### 基本用法

使用默认配置生成可视化：

```bash
python tool/visualize_grid_and_uav_points.py
```

### 指定配置文件

```bash
python tool/visualize_grid_and_uav_points.py --p_yaml trainer_depends/configs/stage3_metric_net.yaml
```

### 完整参数示例

```bash
python tool/visualize_grid_and_uav_points.py \
    --p_yaml trainer_depends/configs/stage3_metric_net.yaml \
    --save_dir vis_results \
    --dpi 300 \
    --max_train 1000 \
    --max_test 500 \
    --show_heatmap
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--p_yaml` | str | `trainer_depends/configs/stage3_metric_net.yaml` | 配置文件路径 |
| `--save_dir` | str | `vis_results` | 图像保存目录 |
| `--dpi` | int | `300` | 图像分辨率（越高越清晰，文件越大） |
| `--max_train` | int | `1000` | 显示的最大训练点数（太多会很慢） |
| `--max_test` | int | `500` | 显示的最大测试点数 |
| `--show_heatmap` | flag | `False` | 是否额外生成热力图 |

## 输出文件

脚本会在 `save_dir` 目录下生成以下文件：

1. **grid_and_uav_points.png**：主要的可视化图像
   - 显示网格划分
   - 训练集位点（蓝色）
   - 测试集位点（红色）
   - 包含统计信息

2. **grid_heatmap.png**（如果使用 `--show_heatmap`）：
   - 左图：训练集密度热力图
   - 右图：测试集密度热力图

## 可视化示例

### 散点图说明

- **网格线**：灰色虚线，显示空间划分
- **边界**：黑色实线，标记地图边界
- **训练点**：蓝色圆点（⚫）
- **测试点**：红色三角（🔺）
- **信息框**：左上角显示网格统计信息

### 图像特点

- **高分辨率**：默认300 DPI，适合论文和报告
- **矢量化网格**：网格线清晰，缩放不失真
- **压缩显示**：自动采样大数据集，避免过度绘制
- **信息丰富**：包含网格大小、坐标范围、点数统计等

## 自定义网格参数

脚本会从配置文件中读取 `n_coarse` 参数。如果需要修改网格划分：

1. 在配置文件中设置：
```yaml
n_coarse: [40, 30, 12, 1]  # [NR, NC, Rot, Scale]
```

2. 或在代码中直接修改（仅用于测试）：
```python
n_coarse = (40, 30, 12, 1)  # 自定义网格大小
```

## 技术细节

### 坐标系统

- **NR (Normalized Row)**：归一化的行坐标，范围 `[nr2sample_min, nr2sample_max]`
- **NC (Normalized Column)**：归一化的列坐标，范围 `[nc2sample_min, nc2sample_max]`
- 坐标已相对于 `satmap_hw_max` 进行归一化

### 网格计算

- **网格数量**：`n_coarse[0]` × `n_coarse[1]` = 总格子数
- **格子大小**：
  - NR方向：`(nr_max - nr_min) / n_coarse[0]`
  - NC方向：`(nc_max - nc_min) / n_coarse[1]`

### 性能优化

- 对于大型数据集，自动随机采样以加快绘制速度
- 使用 LineCollection 批量绘制网格线，提高效率
- 支持自定义采样数量控制内存使用

## 故障排查

### 常见问题

1. **ImportError: No module named ...**
   - 确保在项目根目录下运行
   - 检查是否安装了所需依赖：matplotlib, numpy, torch

2. **数据集加载失败**
   - 检查配置文件中的路径是否正确
   - 确认 `p_satinfo_json`, `p_uav_geocsv_train`, `p_uav_geocsv_test` 路径存在

3. **图像为空或显示异常**
   - 检查坐标范围是否正确
   - 尝试增加 `max_train` 和 `max_test` 参数

## 进阶用法

### 在Python代码中调用

```python
from tool.visualize_grid_and_uav_points import visualize_grid_and_points
from trainer_depends.datasets.dataset_wingtra_4d import SatDataset, UAVDataset

# 加载数据集
sat_dataset = SatDataset(...)
uav_train = UAVDataset(...)
uav_test = UAVDataset(...)

# 生成可视化
fig, ax = visualize_grid_and_points(
    sat_dataset=sat_dataset,
    uav_dataset_train=uav_train,
    uav_dataset_test=uav_test,
    n_coarse=(40, 30, 12, 1),
    save_path='my_visualization.png',
    dpi=300
)
```

### 自定义样式

修改脚本中的绘图参数以自定义外观：

```python
# 修改颜色
ax.scatter(..., c='green', ...)  # 更改训练集颜色

# 修改点大小
ax.scatter(..., s=20, ...)  # 增大点的尺寸

# 修改透明度
ax.scatter(..., alpha=0.8, ...)  # 调整透明度
```

## 更新日志

- **v1.0** (2026-01-04)
  - 初始版本
  - 支持2D网格可视化
  - 支持训练/测试集位点显示
  - 支持热力图生成
