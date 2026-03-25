import os
import sys
import torch
import torch.nn.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# 确保导入正确的 Trainer 类
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainers.stage2_INGP import GridHashFitTrainer

def analyze_grid_rotation_sensitivity(trainer, sample_indices=None, num_angles=72, save_dir='./analysis_grid_rot'):
    """
    分析 Stage 2 (INGP Grid) 对旋转的敏感程度。

    逻辑：
    1. 取一个测试样本的真值坐标 (Ground Truth Coords).
    2. 保持 RC 和 Scale 不变，只改变 Rotation 维度 (-180 到 180 度).
    3. 查询 Grid 在这些坐标下的特征。
    4. 计算这些 Grid 特征与 真值处视觉特征 (Visual Feature @ GT) 的相似度。

    预期结果：
    - 理想 Grid：曲线形状应与 Stage 1 的视觉相似度曲线高度一致（单峰）。
    - 糟糕 Grid：曲线平坦（说明 MLP 忽略了旋转条件，只记住了位置）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 切换到评估模式
    trainer.grid.eval()
    trainer.grid_mlp.eval()
    trainer.vis_encoder.eval()
    trainer.vis_aggregator.eval()

    # 获取数据集
    dataset_uav = trainer.uav_dataset_test

    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset_uav), 3, replace=False)

    print(f"🚀 开始 Grid 旋转敏感性分析，样本ID: {sample_indices}")

    for idx in sample_indices:
        # 1. 获取测试样本数据
        try:
            uav_img, coords_gt = dataset_uav[idx]  # coords: [4]
        except Exception as e:
            print(f"跳过样本 {idx}: {e}")
            continue

        uav_img = uav_img.unsqueeze(0).to(trainer.device)  # [1, C, H, W]
        coords_gt = coords_gt.to(trainer.device).unsqueeze(0)  # [1, 4]

        # 2. 提取真值视觉特征 (作为 Anchor)
        # 我们希望 Grid 在 GT 位置生成的特征能逼近这个视觉特征
        with torch.no_grad():
            feat_vis_gt = trainer._get_feats_fm_imgs(uav_img)  # [1, D]
            feat_vis_gt = TF.normalize(feat_vis_gt, dim=-1)

        # 3. 构造旋转扫描坐标
        # 保持 nr, nc, scale 不变，只变 rotation
        angles_deg = np.linspace(-180, 180, num_angles, endpoint=False)
        angles_rad = np.deg2rad(angles_deg)

        # 复制 GT 坐标 num_angles 份
        coords_scan = coords_gt.repeat(num_angles, 1)  # [N, 4]

        # 替换旋转列 (第3列，索引2)
        # 注意：dataset 返回的 coords_gt[2] 是 GT 旋转
        # 我们现在要扫描的是相对旋转？还是绝对旋转？
        # Stage 1 测试的是“相对旋转”，这里我们也模拟“相对旋转”
        # 即：测试 GT_Rot + Delta_Rot

        gt_rot_rad = coords_gt[0, 2].item()
        scan_rots_rad = gt_rot_rad + angles_rad
        # 归一化到 [-pi, pi] (虽然 coord_normer 会处理，但为了严谨)
        scan_rots_rad = (scan_rots_rad + np.pi) % (2 * np.pi) - np.pi

        coords_scan[:, 2] = torch.tensor(scan_rots_rad).to(trainer.device)

        # 4. 查询 Grid 特征
        sims_grid = []
        sims_vis = []  # 同时计算视觉特征的相似度作为对比（Upper Bound）

        with torch.no_grad():
            # --- A. Grid 查询 ---
            # 1. 坐标归一化
            coords_6d = trainer.coord_normer.raw_to_norm(coords_scan, append_linear_rot=True)

            # 2. Grid 输入 (XYZ)
            grid_in = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
            feats_raw = trainer._get_feats_fm_grid(grid_in)

            # 3. Pos Encoding (Condition)
            cond = trainer.pos_encoder_grid(coords_6d[:, :5])

            # 4. MLP
            feats_grid = trainer.grid_mlp(feats_raw, cond)
            feats_grid = TF.normalize(feats_grid, dim=-1)

            # 5. 计算 Grid Sim (Grid Feat vs GT Vis Feat)
            # [N, D] * [1, D]^T -> [N, 1]
            sims_grid = torch.mm(feats_grid, feat_vis_gt.T).squeeze().cpu().numpy()

            # --- B. 视觉特征 Upper Bound (可选，用于对比) ---
            # 为了验证 Grid 是否学到了视觉特征的趋势，我们最好也画出 Visual Encoder 的曲线
            # 这需要生成旋转的 Sat 图
            # 获取 GT Sat Crop
            sat_img_base = trainer.sat_dataset.crop_satimg_by_4d_coords(coords_gt[0], apply_rotation=True).to(
                trainer.device)

            # 批量旋转 Sat 图
            # 注意：batch_rotate_images_per_sample 接受的是 deg
            # 我们要旋转的是相对角度 angles_deg
            from trainer_depends.utils.util_batch_rotation import batch_rotate_images_per_sample

            # 由于显存限制，分批处理视觉特征
            batch_size = 32
            sims_vis_list = []

            for i in range(0, num_angles, batch_size):
                batch_angles = angles_deg[i: i + batch_size]
                # 复制 base img
                batch_imgs = sat_img_base.unsqueeze(0).repeat(len(batch_angles), 1, 1, 1)  # [B, C, H, W]
                # 旋转
                batch_imgs_rot = batch_rotate_images_per_sample(batch_imgs, batch_angles)

                # 提特征
                feats_v_rot = trainer._get_feats_fm_imgs(batch_imgs_rot)
                feats_v_rot = TF.normalize(feats_v_rot, dim=-1)

                # Sim
                s = torch.mm(feats_v_rot, feat_vis_gt.T).squeeze().cpu().numpy()
                # Handle scalar output for batch_size=1
                if s.ndim == 0: s = np.array([s])
                sims_vis_list.append(s)

            sims_vis = np.concatenate(sims_vis_list)

        # 5. 绘图
        plot_comparison_curve(angles_deg, sims_grid, sims_vis, idx, save_dir)


def plot_comparison_curve(angles, sims_grid, sims_vis, idx, save_dir):
    """绘制 Grid vs Visual 对比曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制视觉特征曲线 (Ground Truth Trend)
    ax.plot(angles, sims_vis, 'b-', alpha=0.5, linewidth=2, label='Visual Encoder (Teacher)')

    # 绘制 Grid 特征曲线 (Learned Field)
    ax.plot(angles, sims_grid, 'r-', linewidth=2, label='INGP Grid (Student)')

    # 标记 GT 0度
    ax.axvline(0, color='g', linestyle='--', label='GT Rotation')

    # 寻找峰值
    max_idx_grid = np.argmax(sims_grid)
    max_idx_vis = np.argmax(sims_vis)

    ax.plot(angles[max_idx_grid], sims_grid[max_idx_grid], 'ro')
    ax.plot(angles[max_idx_vis], sims_vis[max_idx_vis], 'bo')

    ax.set_xlabel('Relative Rotation (deg)')
    ax.set_ylabel('Cosine Similarity to GT')
    ax.set_title(f'Rotation Sensitivity: Grid vs Visual (Sample #{idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'grid_rot_sensitivity_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 对比图已保存: {save_path}")


if __name__ == "__main__":
    # 配置区
    # 请指向 Stage 2 训练好的 opts.yaml (里面包含了 load_stage1_ckpt 的信息)
    stage2_opts_path = '/trainer_depends/configs/stage2_INGPfit.yaml'

    # 模拟命令行
    if '--p_yaml' not in sys.argv:
        sys.argv.extend(['--p_yaml', stage2_opts_path])

    print("🚀 初始化 Stage 2 Trainer...")
    trainer = GridHashFitTrainer()

    # 初始化数据集
    print("📚 初始化数据集...")
    trainer._init_datasets()

    # 初始化坐标归一化器（必须在数据集初始化之后）
    print("📐 初始化坐标归一化器...")
    from trainer_depends.datasets.util_coords_4d_to_euc5d import CoordsNormProcessor
    trainer.coord_normer = CoordsNormProcessor(trainer.sat_dataset)

    # 这一步非常重要：加载测试权重！
    # Trainer 内部的 _load_checkpoints_for_test 会自动加载 Stage 1 和 Stage 2 的权重
    # 但是我们需要确保它能找到文件
    print("📦 加载权重...")
    trainer._load_checkpoints_for_test()

    # 运行分析
    target_samples = [0, 57, 115]  # 建议和 Stage 1 测试同样的样本以便对比
    valid_samples = [i for i in target_samples if i < len(trainer.uav_dataset_test)]
    if not valid_samples: valid_samples = None

    analyze_grid_rotation_sensitivity(
        trainer,
        sample_indices=valid_samples,
        num_angles=72,
        save_dir='/home/data/zwk/pyproj_neuloc_v0/trainers/vis_results'
    )
