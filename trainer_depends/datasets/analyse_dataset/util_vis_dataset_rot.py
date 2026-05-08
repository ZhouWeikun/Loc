import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader

# 导入你的数据集类
from dataset_wingtra_4d import SatDataset, UAVDataset


def check_data_alignment():
    print("🚀 开始初始化数据集...")

    # 1. 初始化数据集 (使用 dataset_wingtra_4d.py 中的默认路径)
    # 请确保这些路径在你当前环境下是可访问的
    sat_json_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json'
    uav_csv_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv'
    uav_json_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json'

    # 初始化 SatDataset
    sat_dataset = SatDataset(
        p_satinfo_json=sat_json_path,
        p_uav_geocsv=uav_csv_path,
        imgsize2net=256,  # 稍微大一点以便观察
        scale_ref_m=200,
    )

    # 初始化 UAVDataset (注意设置 stage='test' 以获取测试集数据)
    uav_dataset = UAVDataset(
        p_uavinfo_json=uav_json_path,
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        scale_ref_m=200,
        stage='test',  # 重点：我们只关心测试集的分布
        use_augmentation=False
    )

    print(f"测试集样本数: {len(uav_dataset)}")

    # 2. 统计 GT 角度分布
    gt_angles_deg = []

    # 3. 抽取部分样本进行可视化对比 (UAV vs Sat Crop)
    vis_indices = np.linspace(0, len(uav_dataset) - 1, 5, dtype=int)  # 随机抽5张均匀分布的
    vis_samples = []

    print("正在遍历测试集收集角度信息...")
    # 为了速度，我们只遍历数据收集角度，不进行耗时的Crop操作，除非是需要可视化的样本
    for i in tqdm.trange(len(uav_dataset)):
        # 获取 UAV 数据
        # UAVDataset 返回: uavimg_q, coords (coords包含 [nrc, rot, scale])
        uav_img, coords = uav_dataset[i]

        # 提取角度 (弧度 -> 度)
        rot_rad = coords[2].item()
        rot_deg = np.rad2deg(rot_rad)
        gt_angles_deg.append(rot_deg)

        # 如果是选中的可视化样本，进行卫星图 Crop
        if i in vis_indices:
            # 使用 GT 坐标 Crop 卫星图
            # 注意：crop_satimg_by_4d_coords 会根据传入的 rot 进行旋转
            # 如果坐标系一致，返回的 sat_crop 应该与 uav_img 方向一致
            sat_crop = sat_dataset.crop_satimg_by_4d_coords(coords, apply_rotation=True)

            vis_samples.append({
                'id': i,
                'uav_img': uav_img,
                'sat_crop': sat_crop,
                'rot_deg': rot_deg
            })

    gt_angles_deg = np.array(gt_angles_deg)

    # --- 绘图部分 ---

    # 图1：角度分布统计 (验证是否真的缺失了某一块)
    plt.figure(figsize=(15, 6))

    # 线性直方图
    plt.subplot(1, 2, 1)
    plt.hist(gt_angles_deg, bins=72, range=(-180, 180), color='green', alpha=0.7)
    plt.title(f'GT Angle Distribution (Linear)\nRange: [{gt_angles_deg.min():.1f}, {gt_angles_deg.max():.1f}]')
    plt.xlabel('Angle (Degrees, -180 to 180)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # 极坐标图 (最直观)
    ax = plt.subplot(1, 2, 2, projection='polar')
    # 将角度转为 0-360 用于极坐标
    angles_360 = (gt_angles_deg % 360 + 360) % 360
    bins_rad = np.linspace(0.0, 2 * np.pi, 37)
    hist, _ = np.histogram(np.deg2rad(angles_360), bins=bins_rad)
    ax.bar(bins_rad[:-1], hist, width=(2 * np.pi) / 36, bottom=0.0, color='green', alpha=0.6)
    ax.set_theta_zero_location('N')  # 0度朝北
    ax.set_theta_direction(-1)  # 顺时针
    plt.title('GT Angle Distribution (Polar View)')

    plt.tight_layout()
    plt.savefig('/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results/test_dataset_distribution.png')
    print("分布图已保存至 test_dataset_distribution.png")
    # plt.show()

    # 图2：可视化对齐检查 (UAV vs Rotated Sat)
    # 这张图能一眼看出是否存在 90度/75度 的系统偏差
    fig, axes = plt.subplots(len(vis_samples), 2, figsize=(8, 4 * len(vis_samples)))
    if len(vis_samples) == 1: axes = axes[None, :]

    for idx, item in enumerate(vis_samples):
        # 反归一化图像以便显示
        uav_vis = uav_dataset.denormalize_img(item['uav_img'])
        sat_vis = sat_dataset.denormalize_img(item['sat_crop'])

        rot_val = item['rot_deg']

        axes[idx, 0].imshow(uav_vis)
        axes[idx, 0].set_title(f"UAV Sample {item['id']}\nGT Rot: {rot_val:.1f}°")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(sat_vis)
        axes[idx, 1].set_title(f"Sat Crop (Rotated by GT)\nShould match UAV orientation")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig('/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results/test_dataset_visual_alignment.png')
    print("可视化对比图已保存至 test_dataset_visual_alignment.png")
    # plt.show()


if __name__ == '__main__':
    check_data_alignment()