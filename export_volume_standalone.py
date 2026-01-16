"""
Export rendered volume from trained Gaussian model for WebGL viewing.
Standalone script that doesn't require CUDA compilation of simple_knn.
"""
import torch
import json
import numpy as np
import os

# Load checkpoint
checkpoint_path = "outputs/gaussian_swc/tiff/checkpoints/model_iter_3000.pth"
config_path = "configs/gaussian_swc.yaml"

# Load config
import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def build_rotation(r):
    """Build rotation matrix from quaternion"""
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cuda')
xyz = checkpoint['xyz'].cuda()
intensity_raw = checkpoint['intensity'].cuda()
scaling_raw = checkpoint['scaling'].cuda()
rotation_raw = checkpoint['rotation'].cuda()

# Apply activations (same as GaussianModel)
intensity = torch.sigmoid(intensity_raw)
scaling = torch.exp(scaling_raw)
rotation = torch.nn.functional.normalize(rotation_raw, dim=-1)

num_gaussians = xyz.shape[0]
print(f"Loaded {num_gaussians} Gaussians")
print(f"XYZ range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
print(f"Intensity range: [{intensity.min().item():.3f}, {intensity.max().item():.3f}]")
print(f"Scale range: [{scaling.min().item():.3f}, {scaling.max().item():.3f}]")

# Build inverse covariance matrices
R = build_rotation(rotation)  # [N, 3, 3]
R_T = R.transpose(1, 2)
scaling_inv_sq = 1.0 / (scaling ** 2)  # [N, 3]
S_inv_sq = torch.diag_embed(scaling_inv_sq)  # [N, 3, 3]
inv_cov = torch.matmul(R, torch.matmul(S_inv_sq, R_T))  # [N, 3, 3]

# Volume dimensions
volume_size = config['img_size']
if isinstance(volume_size, int):
    volume_size = [volume_size, volume_size, volume_size]

Z, Y, X = volume_size[0], volume_size[1], volume_size[2]
print(f"Rendering volume: Z={Z}, Y={Y}, X={X}")

# For memory efficiency, render in chunks
# Downsample for faster rendering
downsample = 2
Z_ds, Y_ds, X_ds = Z // downsample, Y // downsample, X // downsample
print(f"Downsampled to: Z={Z_ds}, Y={Y_ds}, X={X_ds}")

# Create grid
z_coords = torch.linspace(-1, 1, Z_ds, device='cuda')
y_coords = torch.linspace(-1, 1, Y_ds, device='cuda')
x_coords = torch.linspace(-1, 1, X_ds, device='cuda')

# Initialize output volume
volume = torch.zeros(Z_ds, Y_ds, X_ds, device='cuda')

# Render slice by slice for memory efficiency
print("Rendering volume...")
for zi, z_val in enumerate(z_coords):
    if zi % 10 == 0:
        print(f"  Slice {zi}/{Z_ds}")
    
    # Create grid for this Z slice
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    zz = torch.full_like(xx, z_val)
    grid_slice = torch.stack([xx, yy, zz], dim=-1)  # [Y, X, 3]
    grid_flat = grid_slice.reshape(-1, 3)  # [Y*X, 3]
    
    # Process grid points in chunks, Gaussians all at once
    chunk_size = 5000
    slice_intensity = torch.zeros(Y_ds * X_ds, device='cuda')
    
    for start in range(0, Y_ds * X_ds, chunk_size):
        end = min(start + chunk_size, Y_ds * X_ds)
        grid_chunk = grid_flat[start:end]  # [chunk, 3]
        
        # Compute distance from each grid point to each Gaussian
        d = grid_chunk.unsqueeze(1) - xyz.unsqueeze(0)  # [chunk, N, 3]
        
        # Mahalanobis: d^T @ inv_cov @ d
        d_col = d.unsqueeze(-1)  # [chunk, N, 3, 1]
        d_row = d.unsqueeze(-2)  # [chunk, N, 1, 3]
        mahal = torch.matmul(d_row, torch.matmul(inv_cov.unsqueeze(0), d_col))
        mahal = mahal.squeeze(-1).squeeze(-1)  # [chunk, N]
        
        # Gaussian weight
        weight = torch.exp(-0.5 * mahal)  # [chunk, N]
        
        # Intensity contribution (sum over all Gaussians)
        contrib = (weight * intensity.view(1, -1)).sum(dim=1)  # [chunk]
        slice_intensity[start:end] = contrib
    
    volume[zi] = slice_intensity.reshape(Y_ds, X_ds)

print("Rendering complete!")
print(f"Volume range: [{volume.min().item():.4f}, {volume.max().item():.4f}]")

# Convert to numpy and normalize
volume_np = volume.cpu().numpy()
volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

# Save MIP projections
print("Generating MIP projections...")
mip_z = volume_np.max(axis=0)  # Top view
mip_y = volume_np.max(axis=1)  # Front view  
mip_x = volume_np.max(axis=2)  # Side view

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mip_z, cmap='gray')
axes[0].set_title(f'MIP Z (top) - {mip_z.shape}')
axes[1].imshow(mip_y, cmap='gray')
axes[1].set_title(f'MIP Y (front) - {mip_y.shape}')
axes[2].imshow(mip_x, cmap='gray')
axes[2].set_title(f'MIP X (side) - {mip_x.shape}')
plt.tight_layout()
plt.savefig('mip_from_gaussians.png', dpi=150)
print("Saved: mip_from_gaussians.png")

# Export volume for WebGL
print("Exporting volume data...")
export_data = {
    'volume_size': {'x': X, 'y': Y, 'z': Z},
    'rendered_size': {'x': X_ds, 'y': Y_ds, 'z': Z_ds},
    'volume_data': volume_np.flatten().tolist()
}

with open('rendered_volume.json', 'w') as f:
    json.dump(export_data, f)

print(f"Saved: rendered_volume.json ({os.path.getsize('rendered_volume.json') / 1024 / 1024:.1f} MB)")

# Also save numpy
np.save('rendered_volume.npy', volume_np)
print("Saved: rendered_volume.npy")
