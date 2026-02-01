# 快速开始 - 地图网格可视化

## 1分钟快速开始

只需一条命令即可生成可视化图像：

```bash
cd /home/data/zwk/pyproj_neuloc_v0
python tool/visualize_grid_and_uav_points.py
```

这将生成 `vis_results/grid_and_uav_points.png`

## 生成热力图

如果想同时生成热力图，添加 `--show_heatmap` 参数：

```bash
python tool/visualize_grid_and_uav_points.py --show_heatmap
```

## 使用自己的配置文件

如果你想使用不同的实验配置：

```bash
# 使用训练配置
python tool/visualize_grid_and_uav_points.py \
    --p_yaml trainer_depends/configs/stage3_metric_net.yaml

# 使用已有实验的配置
python tool/visualize_grid_and_uav_points.py \
    --p_yaml trainers/.exps/stage3_metric_net_31/opts.yaml
```

## 调整图像质量

### 高质量图像（适合论文）

```bash
python tool/visualize_grid_and_uav_points.py \
    --dpi 600 \
    --max_train 2000 \
    --max_test 1000
```

### 快速预览（低分辨率）

```bash
python tool/visualize_grid_and_uav_points.py \
    --dpi 150 \
    --max_train 500 \
    --max_test 200
```

## 自定义保存路径

```bash
python tool/visualize_grid_and_uav_points.py \
    --save_dir my_visualizations
```

输出将保存到 `my_visualizations/grid_and_uav_points.png`

## 输出示例

### 1. 主可视化图（grid_and_uav_points.png）

包含以下元素：
- ✅ 灰色虚线网格（40×30格）
- ✅ 黑色边界框
- ✅ 蓝色圆点 = 训练集位点
- ✅ 红色三角 = 测试集位点
- ✅ 统计信息框（左上角）

### 2. 热力图（grid_heatmap.png，需要 --show_heatmap）

左右两张子图：
- ✅ 左图：训练集密度分布（蓝色）
- ✅ 右图：测试集密度分布（红色）
- ✅ 颜色越深 = 位点越密集

## 常见问题

### Q: 图像太小，看不清？
A: 增加 `--dpi` 参数，例如 `--dpi 600`

### Q: 点太多，图像很乱？
A: 减少显示的点数，例如 `--max_train 500 --max_test 200`

### Q: 想要显示所有点？
A: 设置很大的数值，例如 `--max_train 99999 --max_test 99999`（注意：会很慢）

### Q: 报错找不到配置文件？
A: 检查 `--p_yaml` 路径是否正确，确保文件存在

### Q: 报错找不到数据集？
A: 检查配置文件中的以下路径是否正确：
- `p_satinfo_json`
- `p_uavinfo_json`
- `p_uav_geocsv`

## 高级技巧

### 1. 批量生成多个实验的可视化

```bash
for exp in trainers/.exps/stage3_*; do
    python tool/visualize_grid_and_uav_points.py \
        --p_yaml $exp/opts.yaml \
        --save_dir $exp/visualizations
done
```

### 2. 在Jupyter Notebook中使用

```python
from tool.visualize_grid_and_uav_points import visualize_grid_and_points
from trainer_depends.datasets.dataset_wingtra_4d import SatDataset, UAVDataset

# 初始化数据集...
sat_dataset = SatDataset(...)
uav_train = UAVDataset(...)
uav_test = UAVDataset(...)

# 生成可视化
fig, ax = visualize_grid_and_points(
    sat_dataset=sat_dataset,
    uav_dataset_train=uav_train,
    uav_dataset_test=uav_test,
    n_coarse=(40, 30, 12, 1),
    save_path='my_plot.png'
)

# 可以继续在同一个figure上绘图
ax.set_title("My Custom Title")
fig.savefig('modified_plot.png')
```

### 3. 只生成热力图

```python
from tool.visualize_grid_and_uav_points import visualize_with_heatmap

fig, axes = visualize_with_heatmap(
    sat_dataset=sat_dataset,
    uav_dataset_train=uav_train,
    uav_dataset_test=uav_test,
    n_coarse=(40, 30, 12, 1),
    save_path='heatmap_only.png'
)
```

## 完整命令示例

```bash
# 最常用的命令
python tool/visualize_grid_and_uav_points.py \
    --p_yaml trainer_depends/configs/stage3_metric_net.yaml \
    --save_dir vis_results \
    --dpi 300 \
    --max_train 1000 \
    --max_test 500 \
    --show_heatmap
```

这会生成两个文件：
1. `vis_results/grid_and_uav_points.png`
2. `vis_results/grid_heatmap.png`

## 下一步

- 查看详细文档：`tool/README_visualization.md`
- 修改脚本自定义样式：编辑 `tool/visualize_grid_and_uav_points.py`
- 集成到你的pipeline中
