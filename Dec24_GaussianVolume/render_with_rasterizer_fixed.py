#!/usr/bin/env python3
"""
Render Volume using Diff-Gaussian-Rasterization (FIXED)

Renders 2D projections of the trained Gaussian model from different viewpoints.
"""

import argparse
import os
import sys
import math
import torch
import numpy as np
import tifffile as tiff
from PIL import Image

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_gs'))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_model_cuda import CUDAGaussianModel


def getWorld2View(R, t):
    """Get world to view transformation matrix (simple version)."""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Get projection matrix."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def look_at(cam_pos, target, up=np.array([0, 1, 0])):
    """Create rotation matrix for camera looking at target.
    
    Uses 3DGS convention where z is positive for points in front of camera.
    """
    # Camera z-axis: forward direction (from camera toward target)
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    # Camera x-axis: right
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    # Camera y-axis: recomputed up
    up_new = np.cross(right, forward)
    
    # Rotation matrix (flip forward for view transform convention)
    R = np.stack([right, up_new, -forward], axis=0).astype(np.float32)
    
    # Translation: world origin position in camera space
    t = R @ cam_pos  # Positive z for points in front
    
    return R, t


def render_view(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cam_pos: np.ndarray,
    target: np.ndarray,
    width: int = 512,
    height: int = 512,
    fov_deg: float = 60.0,
    bg_color: torch.Tensor = None
):
    """Render a single view."""
    if bg_color is None:
        bg_color = torch.zeros(3, device='cuda')
    
    # Camera matrices
    fov = math.radians(fov_deg)
    znear, zfar = 0.01, 100.0
    
    R, t = look_at(cam_pos, target)
    
    # World to view matrix
    world_view = getWorld2View(R, t)
    world_view_t = torch.tensor(world_view, device='cuda').transpose(0, 1)
    
    # Projection matrix
    proj = getProjectionMatrix(znear, zfar, fov, fov).cuda().transpose(0, 1)
    
    # Full projection
    full_proj = world_view_t @ proj
    
    # Camera center
    cam_center = torch.tensor(cam_pos, device='cuda', dtype=torch.float32)
    
    # Rasterizer settings
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fov / 2),
        tanfovy=math.tan(fov / 2),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_t,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    screenspace = torch.zeros_like(means3D)
    
    # Render
    rendered, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    return rendered.clamp(0, 1), radii


def load_gaussian_model(checkpoint_path: str, volume_shape: tuple, intensity_threshold: float = 0.15) -> dict:
    """Load Gaussian model and extract parameters."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    num_gaussians = state_dict['positions'].shape[0]
    
    model = CUDAGaussianModel(
        num_gaussians=num_gaussians,
        volume_shape=volume_shape,
        device='cuda'
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get parameters
    positions = model.positions.detach()  # (N, 3) in [0,1]
    scales_raw = model.scales.detach()
    rotations = model.rotations.detach()
    intensities = model.intensities.detach()
    
    # Volume shape: (D, H, W) = (100, 647, 813)
    D, H, W = volume_shape
    max_dim = max(D, H, W)
    
    # Scale positions to preserve aspect ratio
    scale_factors = torch.tensor([D / max_dim, H / max_dim, W / max_dim], device='cuda')
    
    # Center positions and apply aspect ratio scaling
    means3D = (positions - 0.5) * scale_factors
    
    print(f"  Volume shape: {volume_shape}")
    print(f"  Scale factors (D/H/W normalized): {scale_factors.cpu().numpy()}")
    
    # Scales - must be positive, also scale by aspect ratio
    # Clamp scales to reasonable range to avoid artifacts
    scales = torch.abs(scales_raw) * scale_factors
    max_scale = 0.05  # Maximum scale to prevent huge blurry Gaussians
    scales = torch.clamp(scales, min=1e-6, max=max_scale)
    
    # Normalize rotations
    rotations = rotations / (rotations.norm(dim=1, keepdim=True) + 1e-8)
    
    # Filter by intensity (stricter threshold to remove noise)
    mask = intensities > intensity_threshold
    means3D = means3D[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    intensities_filt = intensities[mask]
    
    # Opacities - directly proportional to intensity
    # Higher intensity = more opaque
    int_norm = (intensities_filt - intensity_threshold) / (intensities_filt.max() - intensity_threshold + 1e-8)
    int_norm = torch.clamp(int_norm, 0, 1)
    opacities = (int_norm * 0.9 + 0.1).unsqueeze(-1)  # Range [0.1, 1.0]
    
    # Colors from intensities (grayscale)
    colors = int_norm.unsqueeze(-1).expand(-1, 3)
    
    print(f"  Original Gaussians: {num_gaussians}")
    print(f"  Filtered (intensity > {intensity_threshold}): {mask.sum().item()}")
    print(f"  Means3D: [{means3D.min():.4f}, {means3D.max():.4f}]")
    print(f"  Scales (clamped): [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  Opacities: [{opacities.min():.4f}, {opacities.max():.4f}]")
    
    return {
        'means3D': means3D,
        'scales': scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacities
    }


def render_orbit(
    gaussian_params: dict,
    num_views: int = 36,
    image_size: int = 512,
    output_dir: str = 'renders',
    radius: float = 2.0,
    elevation: float = 0.3
):
    """Render orbit around the Gaussians."""
    os.makedirs(output_dir, exist_ok=True)
    
    target = np.array([0.0, 0.0, 0.0])
    bg_color = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    
    print(f"\nRendering {num_views} views (radius={radius})...")
    images = []
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        cam_pos = np.array([
            radius * math.cos(angle),
            elevation,
            radius * math.sin(angle)
        ], dtype=np.float32)
        
        with torch.no_grad():
            rendered, radii = render_view(
                means3D=gaussian_params['means3D'],
                scales=gaussian_params['scales'],
                rotations=gaussian_params['rotations'],
                opacities=gaussian_params['opacities'],
                colors=gaussian_params['colors'],
                cam_pos=cam_pos,
                target=target,
                width=image_size,
                height=image_size,
                fov_deg=60.0,
                bg_color=bg_color
            )
        
        visible = (radii > 0).sum().item()
        if i == 0 or i == num_views // 4:
            print(f"  View {i}: {visible} visible Gaussians, img range [{rendered.min():.3f}, {rendered.max():.3f}]")
        
        img = (rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        images.append(img)
        
        Image.fromarray(img).save(os.path.join(output_dir, f'view_{i:03d}.png'))
    
    print(f"Saved {num_views} renders to {output_dir}")
    
    # Create GIF
    try:
        frames = [Image.fromarray(img) for img in images]
        frames[0].save(
            os.path.join(output_dir, 'orbit.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f"Created orbit.gif")
    except Exception as e:
        print(f"Could not create GIF: {e}")
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Render with Diff-Gaussian-Rasterization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--volume', type=str, required=True)
    parser.add_argument('--output', type=str, default='renders')
    parser.add_argument('--num_views', type=int, default=36)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--radius', type=float, default=2.0, help='Camera orbit radius')
    parser.add_argument('--elevation', type=float, default=0.3, help='Camera Y position')
    parser.add_argument('--intensity_threshold', type=float, default=0.15, help='Minimum intensity to show')
    
    args = parser.parse_args()
    
    print(f"Loading volume shape from: {args.volume}")
    volume = tiff.imread(args.volume)
    volume_shape = tuple(volume.shape)
    print(f"  Volume shape: {volume_shape}")
    
    gaussian_params = load_gaussian_model(args.checkpoint, volume_shape, args.intensity_threshold)
    
    render_orbit(
        gaussian_params=gaussian_params,
        num_views=args.num_views,
        image_size=args.image_size,
        output_dir=args.output,
        radius=args.radius,
        elevation=args.elevation
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
