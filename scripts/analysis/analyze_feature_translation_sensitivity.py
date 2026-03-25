import os
import sys
import torch
import torch.nn.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainers.stage1_visual_encoder import VisualEncoderTrainer


def analyze_translation_sensitivity(trainer, sample_indices=None, range_m=40.0, step_m=2.0,
                                    save_dir='./analysis_trans'):
    """
    分析 Feature Encoder 对平移 (Translation) 的敏感程度。

    逻辑：
    1. 取 UAV Query。
    2. 取 GT 位置的 Sat Crop (Rot=GT, Scale=GT, Pos=GT)。
    3. 保持 Rot 和 Scale 不变，人为在 (Row, Col) 方向上施加偏移。
    4. 绘制 相似度 vs. 偏移距离(米) 的曲线。

    Args:
        range_m: 扫描范围，例如 +/- 40米
        step_m: 扫描步长，例如 每隔 2米 采样一次
    """
    os.makedirs(save_dir, exist_ok=True)

    dataset_uav = trainer.uav_dataset_test
    dataset_sat = trainer.sat_dataset

    trainer.vis_encoder.eval()
    trainer.vis_aggregator.eval()

    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset_uav), 3, replace=False)

    print(f"🚀 开始平移敏感性分析，样本ID: {sample_indices}")
    print(f"   扫描范围: +/- {range_m}m, 步长: {step_m}m")

    # 准备扫描的偏移量 (米)
    offsets_m = np.arange(-range_m, range_m + step_m, step_m)

    # 换算系数：米 -> NRC (Normalized Row/Col)
    # nrc = meters / (hw_max_pix * res_m_per_pix) ???
    # 不，dataset 里有 helper: m_per_nrc = satmap_hw_max * geo_res_m
    m_per_nrc = dataset_sat.satmap_hw_max * dataset_sat.geo_res_m
    nrc_per_m = 1.0 / m_per_nrc

    print(f"   地图分辨率换算: 1 NRC = {m_per_nrc:.2f} m")

    for idx in sample_indices:
        # 1. 获取 UAV Query
        try:
            uav_img, coords = dataset_uav[idx]
        except Exception as e:
            print(e);
            continue

        uav_img = uav_img.unsqueeze(0).to(trainer.device)  # [1, C, H, W]

        # 2. 提取 UAV 特征 (Anchor)
        with torch.no_grad():
            feat_patch_uav = trainer.vis_encoder(uav_img)
            feat_uav = trainer.vis_aggregator(feat_patch_uav)
            feat_uav = TF.normalize(feat_uav, dim=-1)  # [1, D]

        # 3. 开始扫描 (Row方向 和 Col方向 分别扫描)
        sims_row = []
        sims_col = []

        # 基础坐标 (GT)
        nr_gt, nc_gt = coords[0].item(), coords[1].item()

        with torch.no_grad():
            # --- 扫描 Row (南北向) ---
            for off_m in tqdm.tqdm(offsets_m, desc=f"Scanning Row #{idx}", leave=False):
                # 计算新坐标
                off_nrc = off_m * nrc_per_m
                coords_new = coords.clone()
                coords_new[0] = nr_gt + off_nrc  # 修改 Row

                # Crop Sat Img (应用 GT 旋转，只变位置)
                sat_img = dataset_sat.crop_satimg_by_4d_coords(coords_new, apply_rotation=True)
                sat_img = sat_img.unsqueeze(0).to(trainer.device)  # [1, C, H, W]

                # 提特征 & 算相似度
                feat_sat_patch = trainer.vis_encoder(sat_img)
                feat_sat = trainer.vis_aggregator(feat_sat_patch)
                feat_sat = TF.normalize(feat_sat, dim=-1)

                sim = torch.mm(feat_uav, feat_sat.T).item()
                sims_row.append(sim)

            # --- 扫描 Col (东西向) ---
            for off_m in tqdm.tqdm(offsets_m, desc=f"Scanning Col #{idx}", leave=False):
                off_nrc = off_m * nrc_per_m
                coords_new = coords.clone()
                coords_new[1] = nc_gt + off_nrc  # 修改 Col

                # Crop
                sat_img = dataset_sat.crop_satimg_by_4d_coords(coords_new, apply_rotation=True)
                sat_img = sat_img.unsqueeze(0).to(trainer.device)  # [1, C, H, W]

                # Sim
                feat_sat_patch = trainer.vis_encoder(sat_img)
                feat_sat = trainer.vis_aggregator(feat_sat_patch)
                feat_sat = TF.normalize(feat_sat, dim=-1)

                sim = torch.mm(feat_uav, feat_sat.T).item()
                sims_col.append(sim)

        # 4. 绘图
        plot_translation_curve(offsets_m, sims_row, sims_col, idx, save_dir)


def plot_translation_curve(offsets, sims_row, sims_col, idx, save_dir):
    plt.figure(figsize=(8, 6))

    # 绘制两条曲线
    plt.plot(offsets, sims_row, 'b-o', markersize=3, label='Shift along Latitude (Row)')
    plt.plot(offsets, sims_col, 'r-o', markersize=3, label='Shift along Longitude (Col)')

    # 标记中心
    plt.axvline(0, color='g', linestyle='--', label='GT Position')

    # 计算半峰全宽 (FWHM) 或 相似度下降到 0.9*Max 的宽度 -> 衡量"频率"
    # 这里简单打印峰值
    max_sim = max(np.max(sims_row), np.max(sims_col))

    plt.title(f'Translation Sensitivity (Sample #{idx})\nPeak Sim: {max_sim:.3f}')
    plt.xlabel('Shift Distance (meters)')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)  # 统一量程以便对比

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'trans_sensitivity_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 图表已保存: {save_path}")


if __name__ == "__main__":
    # 1. 配置文件路径 (根据你的实际位置修改)
    config_path = '/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage1_visual_encoder.yaml'

    # 2. 权重文件路径 (这是最关键的，填入你训练好的pth路径)
    checkpoint_path = "/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/exps/multiscene_12km2_cl_salad_ep100/epoch095.pth"  # <--- 修改这里！

    # 初始化
    if '--p_yaml' not in sys.argv:
        sys.argv.extend(['--p_yaml', config_path])

    print("🚀 初始化网络组件...")
    # 1. 实例化 Trainer
    # 这会自动调用 _init_networks() 创建 vis_encoder 和 vis_aggregator
    trainer = VisualEncoderTrainer()

    print(f"📂 加载权重: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)

        # 根据 stage1_visual_encoder.py 中的保存逻辑：
        # self._save_checkpoint 保存的是 {**param2optimize, **param2freeze}
        # 所以 checkpoint 字典里直接就有 'vis_encoder' 和 'vis_aggregator' 这两个 key

        if 'vis_encoder' in checkpoint:
            trainer.vis_encoder.load_state_dict(checkpoint['vis_encoder'])
            print("  - vis_encoder loaded")
        else:
            print("⚠️ 警告: checkpoint 中未找到 vis_encoder，可能需要检查 key 名称")

        if 'vis_aggregator' in checkpoint:
            trainer.vis_aggregator.load_state_dict(checkpoint['vis_aggregator'])
            print("  - vis_aggregator loaded")
        else:
            print("⚠️ 警告: checkpoint 中未找到 vis_aggregator")

    else:
        print(f"❌ 错误: 找不到权重文件 {checkpoint_path}")
        print("   (将使用随机初始化权重运行，仅用于测试代码流程)")

    # 2. 初始化数据集
    # Trainer 会根据 config 自动加载 train/test 数据集
    print("📚 初始化数据集...")
    trainer._init_datasets()

    # 运行分析
    # 建议选取之前分析过的同一个样本 (例如 Sample 0), 方便和旋转曲线对比
    analyze_translation_sensitivity(
        trainer,
        sample_indices=[0, 57,115],
        range_m=200.0,  # 扫描前后 50米
        step_m=10.0,  # 每 2米 算一次
        save_dir = '/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results'
    )
