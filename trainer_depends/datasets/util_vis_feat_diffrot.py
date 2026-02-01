import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image


def analyze_rotation_feature_similarity(trainer, dataset_uav, dataset_sat,
                                        sample_idx=None, num_angles=72,
                                        device='cuda', save_path='rot_sim_analysis.png'):
    """
    分析特定位置的特征相似度随旋转角度的变化

    Args:
        trainer: 包含模型的 Trainer 实例 (需包含 vis_encoder 和 vis_aggregator)
        dataset_uav: UAV 数据集
        dataset_sat: 卫星图数据集
        sample_idx: 指定样本索引，如果为 None 则随机选取
        num_angles: 旋转采样的数量 (默认 72，即每 5 度采样一次)
        device: 运行设备
    """
    trainer.vis_encoder.eval()
    trainer.vis_aggregator.eval()

    # 1. 随机选取或指定样本
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(dataset_uav))

    print(f"🔬 分析样本 ID: {sample_idx}")

    # 2. 获取 UAV 图像 (Query)
    # 注意：dataset 返回的是 (img, coords)
    uav_img, coords = dataset_uav[sample_idx]
    uav_img = uav_img.unsqueeze(0).to(device)  # [1, C, H, W]

    # 3. 获取对应的 Satellite 图像 (Reference Base)
    # 我们使用 dataset_sat 的 crop 功能，基于 GT 坐标获取正北朝向的卫星图 crop
    # 注意：这里我们手动控制旋转，所以先获取不带旋转(或GT旋转)的基准图
    # 这里假设 coords 包含了 (nrc, rot, scale)
    # 我们先获取一个 "正对齐" 的卫星图 Crop (即假设 Sat 图旋转 0 度)
    # 为了控制变量，我们直接用 GT 坐标 Crop，但在 Crop 时强制旋转为 0 (正北)
    # 或者是直接使用 GT 对应的 Sat Crop，然后我们手动旋转它

    # 方案：获取 GT 位置的 Sat Crop (且应用 GT 旋转，使其与 UAV 对齐)
    # 这样，在 delta_angle = 0 时，应该是相似度最高的

    # 这里的 coords 是 tensor [4]，包含 (nr, nc, rot_rad, scale)
    # 我们先用这个 coords Crop 出一张 "Perfect Match" 的 Sat 图
    sat_img_base = dataset_sat.crop_satimg_by_4d_coords(coords.unsqueeze(0), apply_rotation=True)
    sat_img_base = sat_img_base.to(device)  # [1, C, H, W]

    # 4. 生成旋转序列并提取特征
    angles = np.linspace(-180, 180, num_angles, endpoint=False)
    similarities = []

    # 提取 UAV 特征 (Query)
    with torch.no_grad():
        # 假设 trainer 有个 helper 函数提取特征，或者直接调用网络
        # feat_uav = trainer._get_feats_fm_imgs(uav_img) # 如果有这个helper
        # 或者手动:
        feat_patch_uav = trainer.vis_encoder(uav_img)
        feat_uav = trainer.vis_aggregator(feat_patch_uav)
        feat_uav = TF.normalize(feat_uav, dim=-1)

    print("正在计算旋转相似度分布...")
    fill_val = sat_img_base.mean(dim=(2, 3)).squeeze(0).tolist()
    with torch.no_grad():
        for angle in tqdm.tqdm(angles):
            # 旋转 Sat 图像
            # 注意：rotate 函数接受角度 (degrees)
            # 我们旋转的是 Base Sat 图 (它已经和 UAV 对齐了)
            # 所以 angle=0 时应该是最高点
            sat_img_rot = TVF.rotate(sat_img_base, angle, fill=fill_val)

            # 提取 Sat 特征
            feat_patch_sat = trainer.vis_encoder(sat_img_rot)
            feat_sat = trainer.vis_aggregator(feat_patch_sat)
            feat_sat = TF.normalize(feat_sat, dim=-1)

            # 计算相似度 (Cosine Similarity)
            sim = torch.mm(feat_uav, feat_sat.T).item()
            similarities.append(sim)

    # 5. 绘图与分析
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(angles, similarities, 'b-', linewidth=2, label='Similarity Curve')
    plt.axvline(0, color='g', linestyle='--', label='Ground Truth Alignment (0°)')

    # 寻找峰值
    max_idx = np.argmax(similarities)
    max_angle = angles[max_idx]
    max_sim = similarities[max_idx]

    plt.plot(max_angle, max_sim, 'ro', label=f'Peak: {max_angle:.1f}° (Sim: {max_sim:.2f})')

    # 标注可能的歧义点 (比如 90 度)
    sim_at_90 = similarities[np.argmin(np.abs(angles - 90))]
    sim_at_minus_90 = similarities[np.argmin(np.abs(angles + 90))]
    plt.plot(90, sim_at_90, 'mo', markersize=5)
    plt.plot(-90, sim_at_minus_90, 'mo', markersize=5)

    plt.title(f'Feature Similarity vs Rotation Angle\nSample {sample_idx}')
    plt.xlabel('Relative Rotation Angle (deg)')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-180, 180)

    # 显示原始图片对比
    # 创建一个小图显示 UAV 和 Sat Base

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 分析完成，结果已保存至 {save_path}")
    plt.show()

from dataset_wingtra_4d import SatDataset,UAVDataset
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    # 1. 配置路径 (保持你原有的路径)
    # 请确保这些路径在当前环境下是正确的
    sat_json_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/zurich_blocks12_proj2056_res03m.json'
    uav_csv_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_geo_corrected_v1.csv'
    uav_json_path = '/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info/uavimgs_metainfo.json'

    print("🚀 初始化数据集...")

    # 2. 初始化 Satellite Dataset
    sat_dataset = SatDataset(
        p_satinfo_json=sat_json_path,
        p_uav_geocsv=uav_csv_path,
        imgsize2net=224,
        scale_ref_m=200,
    )

    # 3. 初始化 UAV Dataset
    # 注意：trans_georc2nrc_func 必须从 sat_dataset 传入，以保证坐标系对齐
    uav_dataset = UAVDataset(
        p_uavinfo_json=uav_json_path,
        trans_georc2nrc_func=sat_dataset.transfrom_georc_to_nrc,
        geo_res_m=0.3,
        scale_ref_m=200,
        stage='test',  # 使用测试模式，关闭数据增强，检查原始对齐情况
        use_augmentation=False
    )

    print(f"✅ 数据集加载完成! Test Set Size: {len(uav_dataset)}")

    # 4. 可视化验证 (Visual Alignment Check)
    # 随机抽取 5 张 UAV 图像，并获取其对应的 Satellite Crop
    # 如果代码正确，右边的图应该和左边的图方向完全一致 (Visual Alignment)

    num_samples = 5
    indices = random.sample(range(len(uav_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    plt.subplots_adjust(hspace=0.3)

    print(f"\n🔬 开始可视化验证 (抽取 {num_samples} 个样本)...")

    for i, idx in enumerate(indices):
        # A. 获取 UAV 数据
        # uav_img: Tensor [C, H, W]
        # coords: Tensor [4] -> [nr, nc, rot_rad, scale]
        uav_img, coords = uav_dataset[idx]

        # B. 获取对应的 Satellite Crop (应用 GT 旋转)
        # 关键点：apply_rotation=True
        sat_crop_aligned = sat_dataset.crop_satimg_by_4d_coords(coords, apply_rotation=True)

        # C. 获取对应的 Satellite Crop (不旋转，作为参考)
        # 这样我们可以看到“正北朝向”长什么样
        coords_no_rot = coords.clone()
        coords_no_rot[2] = 0  # 强制旋转为 0
        sat_crop_north = sat_dataset.crop_satimg_by_4d_coords(coords_no_rot, apply_rotation=True)

        # D. 反归一化以便显示
        vis_uav = uav_dataset.denormalize_img(uav_img)
        vis_sat_aligned = sat_dataset.denormalize_img(sat_crop_aligned)
        vis_sat_north = sat_dataset.denormalize_img(sat_crop_north)

        # E. 提取数值信息
        rot_deg = np.rad2deg(coords[2].item())
        scale_ratio = coords[3].item()

        # F. 绘图
        # Col 1: UAV Image (Query)
        ax_uav = axes[i, 0] if num_samples > 1 else axes[0]
        ax_uav.imshow(vis_uav)
        ax_uav.set_title(f"UAV Query [{idx}]\nGT Rot: {rot_deg:.1f}°", color='blue', fontweight='bold')
        ax_uav.axis('off')

        # Col 2: Aligned Sat Crop (Should match UAV)
        ax_sat = axes[i, 1] if num_samples > 1 else axes[1]
        ax_sat.imshow(vis_sat_aligned)
        ax_sat.set_title(f"Sat Crop (Rotated by GT)\nShould match Left", color='green', fontweight='bold')
        ax_sat.axis('off')

        # Col 3: North Sat Crop (Original Map Orientation)
        ax_north = axes[i, 2] if num_samples > 1 else axes[2]
        ax_north.imshow(vis_sat_north)
        ax_north.set_title(f"Sat Crop (North Oriented)\n0° Reference", color='gray')
        ax_north.axis('off')

    print("📊 绘图完成，请检查弹出的窗口或保存的图片。")
    print("   - 如果第一列和第二列看起来方向一致，说明 Dataset 逻辑正确。")
    print("   - 如果第二列看起来像第一列旋转了90度，说明存在坐标系定义偏差。")

    save_path = '/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results/rot_diff.png'
    plt.savefig(save_path)
    print(f"图片已保存至: {save_path}")
    # plt.show() # 如果在服务器上运行，请注释掉此行
