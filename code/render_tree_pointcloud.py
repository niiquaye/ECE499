# render_tree_pointcloud.py
import torch
import numpy as np
from tree_dataset import get_lidar_rays_from_xyz
from lidarnerf.lidar_field import LiDARNeRFNetwork
import os

def compute_ray_directions(H, W, fov_up=30.0, fov_down=-10.0):
    fov_total = abs(fov_down) + abs(fov_up)
    directions = np.zeros((H, W, 3), dtype=np.float32)

    for h in range(H):
        for w in range(W):
            elev = fov_up - (fov_total * h / H)
            azim = 360.0 * w / W - 180.0

            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)

            x = np.cos(elev_rad) * np.cos(azim_rad)
            y = np.cos(elev_rad) * np.sin(azim_rad)
            z = np.sin(elev_rad)
            directions[h, w] = [x, y, z]

    return directions

def render_dense_point_cloud(model_path, base_xyz_file, H=128, W=1024, output_path="dense_output.xyz"):
    # Load sparse .xyz file just to re-use get_lidar_rays grid shape logic
    base_points = np.loadtxt(base_xyz_file)
    _, _, mask = get_lidar_rays_from_xyz(base_points, H=H, W=W)
    directions = compute_ray_directions(H, W)

    ray_dirs = directions[mask].reshape(-1, 3)
    ray_origins = np.zeros_like(ray_dirs, dtype=np.float32)

    # Load trained model
    model = LiDARNeRFNetwork(
        use_viewdirs=False,
        input_ch=3,
        input_ch_views=0,
        D=4,
        W=128
    ).cuda()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run inference
    with torch.no_grad():
        rays_o = torch.from_numpy(ray_origins).float().cuda()
        rays_d = torch.from_numpy(ray_dirs).float().cuda()

        output = model(rays_o, rays_d)
        distances = output['distance'].cpu().numpy()  # shape: (N,)

    # Reconstruct (x, y, z)
    upsampled_xyz = ray_dirs * distances[:, np.newaxis]

    # Save to .xyz file
    np.savetxt(output_path, upsampled_xyz, fmt="%.6f")
    print(f"Saved upsampled point cloud with {upsampled_xyz.shape[0]} points to: {output_path}")

if __name__ == "__main__":
    render_dense_point_cloud(
        model_path="lidar_nerf_tree.pth",
        base_xyz_file="data/tree_sample.xyz",
        H=128,
        W=1024,
        output_path="dense_output.xyz"
    )
