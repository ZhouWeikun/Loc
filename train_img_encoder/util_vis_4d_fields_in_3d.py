import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from tqdm import tqdm

# --- 1. You need to modify the parameters ---

# Define the 3D grid's range and resolution
# WARNING: 3D grids grow cubically. Start with a lower resolution (e.g., 32 or 48)
grid_resolution = 48
row_min, row_max = 0.0, 1.0
col_min, col_max = 0.0, 1.0
rot_min, rot_max = 0, 2 * np.pi  # Rotation range [0, 360 degrees]

# Define the fixed dimension value
fixed_scale = 0.55

# Define the expected ground truth pose (now optional)
gt_pose_true = (0.228, 0.319, 0.55, np.deg2rad(139.92))

# Define the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Your model, encoders, and features will be passed to the function ---
# The following are MOCK objects for demonstration purposes.
# In your code, you will pass your ACTUAL trained models.

# Mock Positional Encoders (replace with your actual encoders)
mock_rc_encoder = nn.Identity()
mock_rot_encoder = nn.Identity()
mock_scale_encoder = nn.Identity()
mock_pos_encoders = (mock_rc_encoder, mock_rot_encoder, mock_scale_encoder)

# Mock Conditional Feature (replace with your actual feature vector)
# This would come from your image encoder, e.g., c_feat = img_encoder(uav_image)
mock_c_feat = torch.randn(1, 256).to(device)  # Example: 256-dim feature


# Mock Model/Decoder (replace with your actual decoder model)
# It must accept two arguments: encoded poses and conditional features
def mock_decoder_func(encoded_poses, c_features):
    """A mock decoder function for demonstration."""
    # This simple mock doesn't use the features, but shows the required signature
    gt_r, gt_c, gt_s, gt_d = gt_pose_true
    # Simulate a response based on the un-encoded values for simplicity
    # NOTE: This is NOT how the real model works, it's just for a plausible visual
    num_points = encoded_poses.shape[0]
    rc_vals = torch.rand(num_points, 2)  # Cannot easily invert pos encoding
    d_vals = torch.rand(num_points, 1)
    s_vals = torch.full((num_points, 1), fixed_scale)

    gt_rc_tensor = torch.tensor([gt_r, gt_c], device=device)
    gt_d_tensor = torch.tensor([gt_d], device=device)

    dist_rc = torch.norm(rc_vals - gt_rc_tensor, dim=1)
    dist_d = torch.abs(d_vals.squeeze() - gt_d_tensor)

    return (dist_rc + 0.3 * dist_d).unsqueeze(1)


# --- 3. Main visualization function (Refactored) ---

def visualize_udf_3d_rc_rot(r_range, c_range, rot_range, resolution, s_val,
                            model_func, pos_encoders, c_feat, gt_pose=None, batch_size=4096):
    """
    Generates and plots a 3D isosurface for the UDF field's (row, col, rot) space,
    using the same preprocessing pipeline as the 2D function.
    """
    print(f"1. Creating 3D coordinate grid ({resolution}x{resolution}x{resolution})...")
    C, R, ROT = np.mgrid[c_range[0]:c_range[1]:resolution * 1j,
                r_range[0]:r_range[1]:resolution * 1j,
                rot_range[0]:rot_range[1]:resolution * 1j]

    grid_points = np.stack([R.ravel(), C.ravel(), ROT.ravel()], axis=1)
    num_points = len(grid_points)

    print(f"2. Preparing and encoding {num_points} query poses...")
    # Prepare un-encoded values
    rc_vals = torch.from_numpy(grid_points[:, 0:2]).float()
    d_vals = torch.from_numpy(grid_points[:, 2].reshape(-1, 1)).float()
    s_vals = torch.full((num_points, 1), s_val, dtype=torch.float32)

    # Encode poses using the provided encoders
    rc_poses = pos_encoders[0](rc_vals.to(device))
    rot_inputs = torch.cat([torch.sin(d_vals), torch.cos(d_vals)], dim=-1)
    rot_poses = pos_encoders[1](rot_inputs.to(device))
    scale_poses = pos_encoders[2](s_vals.to(device))

    # Concatenate to form the final encoded pose tensor
    all_poses_encoded = torch.cat([rc_poses, rot_poses, scale_poses], dim=-1)

    # Expand the conditional feature to match the number of poses
    c_feat_expanded = c_feat.expand(num_points, -1)

    print("3. Running model inference in batches...")
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, num_points, batch_size)):
            batch_poses = all_poses_encoded[i:i + batch_size]
            batch_c_feat = c_feat_expanded[i:i + batch_size]

            # Call the model with both poses and features
            batch_results = model_func(batch_poses, batch_c_feat)
            results.append(batch_results.cpu())

    predicted_distances = torch.cat(results).numpy().flatten()



    print("4. Plotting the 3D isosurface...")
    fig = go.Figure()

    min_dist = predicted_distances.min()
    max_dist = predicted_distances.max()
    print(f"Predicted distances range: [{min_dist:.4f}, {max_dist:.4f}]")

    fig.add_trace(go.Isosurface(
        x=C.flatten(),
        y=R.flatten(),
        z=np.rad2deg(ROT.flatten()),
        value=predicted_distances,
        isomin=min_dist,
        isomax=min_dist + (max_dist - min_dist) * 0.3,
        surface_count=5,
        opacity=0.2,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title='Pred Distance')
    ))

    # Add ground truth marker if provided
    if gt_pose is not None:
        gt_r, gt_c, gt_d, gt_s = gt_pose
        if abs(gt_s - s_val) < 0.1:
            fig.add_trace(go.Scatter3d(
                x=[gt_c], y=[gt_r], z=[np.rad2deg(gt_d)],
                mode='markers',
                marker=dict(color='red', size=8, symbol='cross'),
                name='GT'
            ))

    # --- 新增逻辑：绘制预测的最小值点 ---
    min_index = np.argmin(predicted_distances)
    min_dist_value = predicted_distances[min_index]
    # 从grid_points中找到对应的 (r, c, rot) 坐标
    pred_r, pred_c, pred_rot = grid_points[min_index]

    fig.add_trace(go.Scatter3d(
        x=[pred_c], y=[pred_r], z=[np.rad2deg(pred_rot)],
        mode='markers',
        marker=dict(color='cyan', size=8, symbol='diamond'),  # 使用青色菱形标记
        # name=f'pred_min_dist: {min_dist_value:.3f})'
        name = 'Pred'
    ))

    # (fig.update_layout 部分与之前相同)
    # fig.update_layout(
    #     title=f'UDF 3D Visualization (Image Coordinate System)',
    #     scene=dict(
    #         xaxis_title='Column (col)',
    #         yaxis=dict(title='Row (row)', autorange='reversed'),
    #         zaxis_title='Rotation (degrees)',
    #         aspectratio=dict(x=1, y=1, z=1)
    #     )
    # )
    fig.update_layout(
        title=f'UDF 3D Visualization (Image Coordinate System)',
        scene=dict(
            xaxis_title='Column (col)',
            yaxis=dict(title='Row (row)', autorange='reversed'),
            zaxis_title='Rotation (degrees)',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        # 将图例移动到左上角
        legend=dict(
            x=0.01,  # 靠近左边缘
            y=0.99,  # 靠近上边缘
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.6)',  # 使用半透明背景
            bordercolor='white',
            borderwidth=1
        )
    )
    fig.show()

    fig.show()



# --- 4. How to call the function ---
if __name__ == '__main__':
    # In your actual code, you would pass your trained models and features
    visualize_udf_3d_rc_rot(
        r_range=(row_min, row_max),
        c_range=(col_min, col_max),
        rot_range=(rot_min, rot_max),
        resolution=grid_resolution,
        s_val=fixed_scale,
        model_func=mock_decoder_func,  # <--- Pass your trained decoder
        pos_encoders=mock_pos_encoders,  # <--- Pass your tuple of encoders
        c_feat=mock_c_feat,  # <--- Pass the feature from your image encoder
        gt_pose=gt_pose_true
    )