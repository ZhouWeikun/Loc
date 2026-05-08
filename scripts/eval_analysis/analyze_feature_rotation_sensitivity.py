import os
import sys
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import numpy as np
import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainers.stage3_legacy.stage1_visual_encoder import VisualEncoderTrainer


def analyze_rotation_sensitivity(trainer, sample_indices=None, num_angles=72, save_dir='./analysis_rot'):
    """
    核心分析函数：计算特征相似度随旋转角度的变化曲线
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 获取数据集 (使用测试集)
    dataset_uav = trainer.uav_dataset_test
    dataset_sat = trainer.sat_dataset

    # 切换到评估模式 (关闭 Dropout/BatchNorm 更新)
    trainer.vis_encoder.eval()
    trainer.vis_aggregator.eval()

    # 如果未指定样本，随机抽取
    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset_uav), 3, replace=False)

    print(f"🚀 开始旋转敏感性分析，样本ID: {sample_indices}")

    for idx in sample_indices:
        # 1. 获取 UAV Query (Anchor)
        try:
            uav_img, coords = dataset_uav[idx]
        except Exception as e:
            print(f"跳过样本 {idx}: {e}")
            continue

        uav_img = uav_img.unsqueeze(0).to(trainer.device)  # [1, C, H, W]

        # 2. 获取基准 Sat Image (Base)
        # 关键步骤：利用 GT 坐标获取一张与 UAV 物理方向一致的卫星图 Crop
        # 此时 Relative Rotation = 0
        sat_img_base = dataset_sat.crop_satimg_by_4d_coords(coords, apply_rotation=True)
        sat_img_base = sat_img_base.unsqueeze(0).to(trainer.device)  # [1, C, H, W]

        # 3. 提取 UAV 特征
        with torch.no_grad():
            feat_patch_uav = trainer.vis_encoder(uav_img)
            feat_uav = trainer.vis_aggregator(feat_patch_uav)
            feat_uav = TF.normalize(feat_uav, dim=-1)  # [1, D]

        # 4. 旋转扫描 (-180 到 180 度)
        angles = np.linspace(-180, 180, num_angles, endpoint=False)
        sims = []

        fill_val = sat_img_base.mean(dim=(2, 3)).squeeze(0).tolist()
        with torch.no_grad():
            for angle in tqdm.tqdm(angles, desc=f"Scanning Sample {idx}", leave=False):
                # 旋转 Sat Base (逆时针旋转 angle 度)
                sat_img_rot = TVF.rotate(sat_img_base, float(angle), fill=fill_val)

                # 提取 Sat 特征
                feat_patch_sat = trainer.vis_encoder(sat_img_rot)
                feat_sat = trainer.vis_aggregator(feat_patch_sat)
                feat_sat = TF.normalize(feat_sat, dim=-1)

                # 计算余弦相似度
                sim = torch.mm(feat_uav, feat_sat.T).item()
                sims.append(sim)

        # 5. 绘图
        plot_rotation_curve(uav_img, sat_img_base, angles, sims, idx, save_dir, trainer)


def plot_rotation_curve(uav_img, sat_img_base, angles, sims, idx, save_dir, trainer):
    """辅助绘图函数：画出 UAV, Sat 和 相似度曲线"""

    # 辅助函数：反归一化图片以便显示
    def denorm(img_tensor):
        mean = torch.tensor(trainer.sat_dataset.satinfo_dict['means_normalized'][0]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor(trainer.sat_dataset.satinfo_dict['stds_normalized'][0]).view(3, 1, 1).to(img_tensor.device)
        img = img_tensor * std + mean
        return img.permute(0, 2, 3, 1).cpu().numpy()[0]

    vis_uav = np.clip(denorm(uav_img), 0, 1)
    vis_sat = np.clip(denorm(sat_img_base), 0, 1)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])

    # 图1：UAV Query
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(vis_uav)
    ax1.set_title(f'UAV Query #{idx}')
    ax1.axis('off')

    # 图2：Sat Base (0° Reference)
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(vis_sat)
    ax2.set_title('Sat Aligned (0° Rel)')
    ax2.axis('off')

    # 图3：相似度曲线
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(angles, sims, 'b-', linewidth=2, label='Similarity')

    # 标记 GT 位置 (0度)
    ax3.axvline(0, color='g', linestyle='--', alpha=0.5)

    # 寻找峰值点
    max_idx = np.argmax(sims)
    max_angle = angles[max_idx]
    max_sim = sims[max_idx]
    ax3.plot(max_angle, max_sim, 'ro', label=f'Peak: {max_angle:.1f}°')

    # 标记关键角度 (检查 90 度歧义)
    for check_angle in [-90, 90, 180]:
        # 找到最接近该角度的点
        chk_idx = np.argmin(np.abs(angles - check_angle))
        chk_sim = sims[chk_idx]
        ax3.plot(angles[chk_idx], chk_sim, 'mo', markersize=4)
        if chk_sim > max_sim * 0.9:  # 如果这个角度的相似度非常高，报警
            ax3.text(angles[chk_idx], chk_sim, ' Ambiguity!', color='red', fontsize=8)

    ax3.set_xlabel('Relative Rotation (deg)')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Rotation Sensitivity Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(min(sims) - 0.1, 1.1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'rot_sensitivity_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 分析图已保存: {save_path}")


if __name__ == "__main__":
    # ================= 配置区 =================
    # 1. 配置文件路径 (根据你的实际位置修改)
    config_path = '/trainer_depends/configs/stage1_visual_encoder.yaml'

    # 2. 权重文件路径 (这是最关键的，填入你训练好的pth路径)
    checkpoint_path = "/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/exps/multiscene_12km2_cl_salad_ep100/epoch095.pth"  # <--- 修改这里！
    # =========================================

    # 模拟命令行参数，以便 BaseTrainer 正确读取 Config
    if '--p_yaml' not in sys.argv:
        sys.argv.extend(['--p_yaml', config_path])
        # 如果需要指定 GPU，也可以在这里加
        # sys.argv.extend(['--gpu_ids', '0'])

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

    # 3. 运行分析
    # 这里我们手动挑选几个样本，或者随机挑
    # 建议选之前看过的有问题的那几个 ID (比如 57, 115)

    
    target_samples = [0, 57, 115]

    # 如果数据集不够大，防止越界
    valid_samples = [i for i in target_samples if i < len(trainer.uav_dataset_test)]
    if not valid_samples:
        valid_samples = None  # 随机采样

    analyze_rotation_sensitivity(
        trainer,
        sample_indices=valid_samples,
        num_angles=72,  # 每5度采样一次
        save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results'
    )
