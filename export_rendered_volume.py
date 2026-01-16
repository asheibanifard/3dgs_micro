"""
Export rendered volume from trained Gaussian model for WebGL viewing.
This exports the actual volumetric rendering (like training) not just Gaussian points.
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

# Setup model
from models.gaussian_model import GaussianModel

class OptimizationParams:
    def __init__(self):
        self.position_lr_init = 0.002
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.intensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

gaussians = GaussianModel()

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
gaussians._xyz = torch.nn.Parameter(checkpoint['xyz'].cuda())
gaussians._intensity = torch.nn.Parameter(checkpoint['intensity'].cuda())
gaussians._scaling = torch.nn.Parameter(checkpoint['scaling'].cuda())
gaussians._rotation = torch.nn.Parameter(checkpoint['rotation'].cuda())

print(f"Loaded {gaussians._xyz.shape[0]} Gaussians")

# Create sampling grid at full resolution
volume_size = config['img_size']
if isinstance(volume_size, int):
    volume_size = [volume_size, volume_size, volume_size]

# Actual volume dimensions (Z, Y, X)
Z, Y, X = volume_size[0], volume_size[1], volume_size[2]
print(f"Rendering volume: Z={Z}, Y={Y}, X={X}")

# Create normalized grid [-1, 1]
z = torch.linspace(-1, 1, Z)
y = torch.linspace(-1, 1, Y) 
x = torch.linspace(-1, 1, X)

# Create meshgrid
zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).cuda()  # [1, Z, Y, X, 3]

print(f"Grid shape: {grid.shape}")

# Render volume using the same method as training
with torch.no_grad():
    rendered_volume = gaussians.grid_sample(grid)  # [1, Z, Y, X, 1]

print(f"Rendered volume shape: {rendered_volume.shape}")
print(f"Value range: [{rendered_volume.min().item():.4f}, {rendered_volume.max().item():.4f}]")

# Convert to numpy
volume_np = rendered_volume.squeeze().cpu().numpy()  # [Z, Y, X]

# Normalize to [0, 1]
volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

# Export for WebGL as slices (Z slices as base64 images or raw data)
# For simplicity, export as JSON with downsampled volume
downsample = 2  # Reduce size for web
volume_small = volume_np[::downsample, ::downsample, ::downsample]

print(f"Downsampled volume shape: {volume_small.shape}")

# Export as JSON
export_data = {
    'volume_size': {'x': X, 'y': Y, 'z': Z},
    'downsampled_size': {'x': volume_small.shape[2], 'y': volume_small.shape[1], 'z': volume_small.shape[0]},
    'downsample_factor': downsample,
    # Flatten volume for JSON (row-major: z, y, x)
    'volume_data': volume_small.flatten().tolist()
}

with open('rendered_volume.json', 'w') as f:
    json.dump(export_data, f)

print(f"Exported to rendered_volume.json")
print(f"File size: {os.path.getsize('rendered_volume.json') / 1024 / 1024:.1f} MB")

# Also save as NumPy for TIFF export
np.save('rendered_volume.npy', volume_np)
print(f"Also saved as rendered_volume.npy")

# Export MIP projections for quick visualization
mip_z = volume_np.max(axis=0)  # Max along Z -> Y x X
mip_y = volume_np.max(axis=1)  # Max along Y -> Z x X  
mip_x = volume_np.max(axis=2)  # Max along X -> Z x Y

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mip_z, cmap='gray')
axes[0].set_title(f'MIP Z (top view) - {mip_z.shape}')
axes[1].imshow(mip_y, cmap='gray')
axes[1].set_title(f'MIP Y (front view) - {mip_y.shape}')
axes[2].imshow(mip_x, cmap='gray')
axes[2].set_title(f'MIP X (side view) - {mip_x.shape}')
plt.tight_layout()
plt.savefig('mip_projections.png', dpi=150)
print("Saved MIP projections to mip_projections.png")
