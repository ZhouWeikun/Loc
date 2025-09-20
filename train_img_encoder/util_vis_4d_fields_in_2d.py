import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. 您需要修改的参数 (Parameters You Need to Modify) ---

# 定义采样范围和分辨率 (Row, Col 约定)
row_min, row_max = 450, 650  # Y轴 (行) 的范围
col_min, col_max = 400, 600  # X轴 (列) 的范围
grid_resolution = 200  # 网格的分辨率 (例如 200x200 个点)

# 定义固定的维度值
fixed_scale = 100.0  # 固定的尺度/高度 s
fixed_rot = np.deg2rad(45.0)  # 固定的方向 d (注意: 模型通常使用弧度)

# 定义我们期望的真值位置 (现在是可选的)
gt_row, gt_col = 540, 512

# 定义计算时使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. 您的模型函数 (Your Model Function) ---
# (这部分保持不变, 您仍需用自己的模型替换)
def mock_udf_func(poses_tensor: torch.Tensor) -> torch.Tensor:
    """
    这是一个示例函数，请用您自己的 F_world 模型替换它。
    为了演示，这个函数模拟了一个以 (gt_row, gt_col) 为中心的理想 "碗状" UDF。
    """
    gt_pose_tensor = torch.tensor([gt_row, gt_col, fixed_scale, fixed_rot],
                                  dtype=poses_tensor.dtype, device=poses_tensor.device)
    weights = torch.tensor([1.0, 1.0, 0.1, 0.1], device=poses_tensor.device)
    distances = torch.sqrt(torch.sum(weights * (poses_tensor - gt_pose_tensor) ** 2, dim=1))
    return distances


udf_func = mock_udf_func


# --- 3. 可视化主函数 (Main Visualization Function with Optional GT) ---

def visualize_udf_slice_rc(row_range, col_range, resolution, s_val, d_val,
                           model_func, pos_encoders, c_feat, gt_rc=None, batch_size=4096):
    """
    为UDF场的 (row, col) 切片生成并绘制热力图。
    真值点 (gt_r, gt_c) 的可视化是可选的。

    Args:
        row_range (tuple): (min, max) for row dimension.
        col_range (tuple): (min, max) for col dimension.
        resolution (int): The resolution of the grid.
        s_val (float): Fixed scale value.
        d_val (float): Fixed rotation value in radians.
        model_func (callable): The model function to evaluate.
        gt_r (float, optional): Ground truth row coordinate. Defaults to None.
        gt_c (float, optional): Ground truth col coordinate. Defaults to None.
        batch_size (int, optional): Batch size for model inference. Defaults to 4096.
    """
    print("1. Creating (row, col) coordinate grid...")
    row_coords = np.linspace(row_range[0], row_range[1], resolution)
    col_coords = np.linspace(col_range[0], col_range[1], resolution)
    col_grid, row_grid = np.meshgrid(col_coords, row_coords)
    grid_points = np.stack([row_grid.ravel(), col_grid.ravel()], axis=1)
    num_points = len(grid_points)

    print(f"2. Preparing {num_points} query poses for the model...")
    rc_vals = torch.from_numpy(grid_points).float()
    d_vals = torch.full((num_points, 1), d_val, dtype=torch.float32)
    s_vals = torch.full((num_points, 1), s_val, dtype=torch.float32)
    rc_poses = pos_encoders[0](rc_vals)
    rot_poses = pos_encoders[1](torch.concatenate( [torch.sin(d_vals),torch.cos(d_vals)],dim=-1))
    scale_poses = pos_encoders[2](s_vals)
    all_poses = torch.concatenate([rc_poses, rot_poses,scale_poses], dim=-1)

    c_feat = c_feat.unsqueeze(0).expand(all_poses.shape[0],-1)

    print("3. Running model inference in batches...")
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, num_points, batch_size)):
            batch_poses = all_poses[i:i + batch_size].to(c_feat.device)
            batch_c_feat = c_feat[i:i + batch_size]
            batch_results = model_func(batch_poses,batch_c_feat)
            results.append(batch_results.cpu())

    predicted_distances = torch.cat(results).numpy()
    # predicted_distances_cliped=np.clip(predicted_distances,a_min=0.,a_max=0.1)
    result_grid = predicted_distances.reshape((resolution, resolution))

    print("4. Plotting the results...")
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(result_grid,
                   origin='upper',  # <--- 改动 1
                   cmap='viridis_r',
                   extent=[col_range[0], col_range[1], row_range[1], row_range[0]])  # <--- 改动 2

    cbar = fig.colorbar(im)
    cbar.set_label('Predicted Distance (Lower is Better)', rotation=270, labelpad=20)

    # --- 这里是关键改动 ---
    # 只有当 gt_r 和 gt_c 都被提供时，才绘制真值点
    if gt_rc is not None:
        gt_r,gt_c= gt_rc[0][0], gt_rc[0][1]
        ax.scatter(gt_c, gt_r, color='red', marker='x', s=100, label=f'Ground Truth ({gt_r}, {gt_c})')
        ax.legend()  # 只有在有GT点时才显示图例

    ax.set_title(f'UDF Visualization\n(Fixed Scale={s_val:.2f}, Fixed Rotation={np.rad2deg(d_val):.2f}°)')
    ax.set_xlabel('Column (col)')
    ax.set_ylabel('Row (row)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()


# --- 4. 执行可视化 (演示两种情况) ---
if __name__ == '__main__':
    # --- 情况1：传入真值点进行可视化 ---
    print("\n--- Visualizing with Ground Truth Marker ---")
    visualize_udf_slice_rc(
        row_range=(row_min, row_max),
        col_range=(col_min, col_max),
        resolution=grid_resolution,
        s_val=fixed_scale,
        d_val=fixed_rot,
        model_func=udf_func,
        gt_r=gt_row,
        gt_c=gt_col
    )

    # --- 情况2：不传入真值点，只观察场的分布 ---
    print("\n--- Visualizing without Ground Truth Marker ---")
    visualize_udf_slice_rc(
        row_range=(row_min, row_max),
        col_range=(col_min, col_max),
        resolution=grid_resolution,
        s_val=fixed_scale,
        d_val=fixed_rot,
        model_func=udf_func
        # 注意：这里没有传入 gt_r 和 gt_c 参数
    )