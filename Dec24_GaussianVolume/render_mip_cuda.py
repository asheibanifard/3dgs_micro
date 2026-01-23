#!/usr/bin/env python3
"""
CUDA MIP (Maximum Intensity Projection) Renderer for Gaussians

Renders MIP projections by evaluating Gaussian contributions along rays
and taking the maximum intensity instead of alpha blending.
"""

import argparse
import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from PIL import Image

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_gs'))

from gaussian_model_cuda import CUDAGaussianModel


@torch.jit.script
def evaluate_gaussians_at_points_chunked(
    points: torch.Tensor,  # (num_points, 3)
    means: torch.Tensor,   # (N, 3)
    scales: torch.Tensor,  # (N, 3)
    intensities: torch.Tensor,  # (N,)
    gaussian_chunk_size: int = 256
) -> torch.Tensor:
    """
    Evaluate Gaussian contributions at sample points.
    Chunked over Gaussians for memory efficiency.
    """
    num_points = points.shape[0]
    N = means.shape[0]
    
    # Output: intensity at each point
    point_intensity = torch.zeros(num_points, device=points.device)
    
    # Process Gaussians in chunks
    for g_start in range(0, N, gaussian_chunk_size):
        g_end = min(g_start + gaussian_chunk_size, N)
        
        chunk_means = means[g_start:g_end]  # (chunk_g, 3)
        chunk_scales = scales[g_start:g_end]  # (chunk_g, 3)
        chunk_int = intensities[g_start:g_end]  # (chunk_g,)
        
        # Compute distances: (num_points, chunk_g, 3)
        diff = points.unsqueeze(1) - chunk_means.unsqueeze(0)
        
        # Mahalanobis distance
        inv_var = 1.0 / (chunk_scales ** 2 + 1e-8)  # (chunk_g, 3)
        dist_sq = (diff ** 2 * inv_var.unsqueeze(0)).sum(dim=-1)  # (num_points, chunk_g)
        
        # Gaussian values weighted by intensity
        gauss = torch.exp(-0.5 * dist_sq) * chunk_int.unsqueeze(0)  # (num_points, chunk_g)
        
        # Sum contribution from this chunk
        point_intensity += gauss.sum(dim=1)
    
    return point_intensity


def render_mip_cuda(
    means: torch.Tensor,
    scales: torch.Tensor,
    intensities: torch.Tensor,
    cam_pos: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0]),
    width: int = 512,
    height: int = 512,
    fov_deg: float = 60.0,
    num_samples: int = 128,
    near: float = 0.1,
    far: float = 3.0
) -> torch.Tensor:
    """
    Render MIP projection using CUDA ray marching.
    Memory-efficient implementation with chunked processing.
    """
    device = means.device
    
    # Build camera rays
    fov = math.radians(fov_deg)
    aspect = width / height
    
    # Camera basis vectors
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_vec = np.cross(right, forward)
    
    # Convert to tensors
    cam_pos_t = torch.tensor(cam_pos, device=device, dtype=torch.float32)
    forward_t = torch.tensor(forward, device=device, dtype=torch.float32)
    right_t = torch.tensor(right, device=device, dtype=torch.float32)
    up_t = torch.tensor(up_vec, device=device, dtype=torch.float32)
    
    # Pixel coordinates
    tan_fov = math.tan(fov / 2)
    
    # Output image
    mip_image = torch.zeros(height, width, device=device)
    
    # Process in row chunks for memory efficiency
    row_chunk = 32
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        chunk_h = row_end - row_start
        
        i = torch.arange(width, device=device, dtype=torch.float32)
        j = torch.arange(row_start, row_end, device=device, dtype=torch.float32)
        ii, jj = torch.meshgrid(i, j, indexing='xy')
        
        # NDC
        px = (2 * (ii + 0.5) / width - 1) * tan_fov * aspect
        py = (1 - 2 * (jj + 0.5) / height) * tan_fov
        
        # Ray directions (width, chunk_h, 3)
        ray_dirs = forward_t + px.unsqueeze(-1) * right_t + py.unsqueeze(-1) * up_t
        ray_dirs = F.normalize(ray_dirs, dim=-1)
        
        # Sample along rays
        t_vals = torch.linspace(near, far, num_samples, device=device)
        
        # Generate sample points (width, chunk_h, num_samples, 3)
        ray_origins = cam_pos_t.reshape(1, 1, 1, 3)
        ray_dirs_exp = ray_dirs.unsqueeze(2)
        t_exp = t_vals.reshape(1, 1, num_samples, 1)
        
        sample_points = ray_origins + ray_dirs_exp * t_exp  # (W, chunk_h, samples, 3)
        
        # Flatten for evaluation
        flat_points = sample_points.reshape(-1, 3)  # (W * chunk_h * samples, 3)
        
        # Evaluate Gaussians
        point_vals = evaluate_gaussians_at_points_chunked(
            flat_points, means, scales, intensities, gaussian_chunk_size=256
        )
        
        # Reshape and take max along ray (MIP)
        point_vals = point_vals.reshape(width, chunk_h, num_samples)
        chunk_mip = point_vals.max(dim=2).values  # (W, chunk_h)
        
        mip_image[row_start:row_end, :] = chunk_mip.T
    
    return mip_image


def load_gaussian_model(checkpoint_path: str, volume_shape: tuple, intensity_threshold: float = 0.15) -> dict:
    """Load Gaussian model for MIP rendering."""
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
    positions = model.positions.detach()
    scales_raw = model.scales.detach()
    intensities = model.intensities.detach()
    
    # Volume shape scaling
    D, H, W = volume_shape
    max_dim = max(D, H, W)
    scale_factors = torch.tensor([D / max_dim, H / max_dim, W / max_dim], device='cuda')
    
    # Center and scale positions
    means = (positions - 0.5) * scale_factors
    
    # Scales - must be positive
    scales = torch.abs(scales_raw) * scale_factors
    scales = torch.clamp(scales, min=1e-4, max=0.1)
    
    # Filter by intensity
    mask = intensities > intensity_threshold
    means = means[mask]
    scales = scales[mask]
    intensities_filt = intensities[mask]
    
    # Normalize intensities for MIP display
    int_norm = (intensities_filt - intensity_threshold) / (intensities_filt.max() - intensity_threshold + 1e-8)
    int_norm = torch.clamp(int_norm, 0, 1)
    
    print(f"  Volume shape: {volume_shape}")
    print(f"  Filtered (intensity > {intensity_threshold}): {mask.sum().item()}")
    print(f"  Means range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"  Scales range: [{scales.min():.6f}, {scales.max():.6f}]")
    
    return {
        'means': means,
        'scales': scales,
        'intensities': int_norm
    }


def render_orbit_mip(
    params: dict,
    num_views: int = 36,
    image_size: int = 512,
    output_dir: str = 'renders_mip',
    radius: float = 2.0,
    elevation: float = 0.2,
    num_samples: int = 128
):
    """Render MIP orbit around the volume."""
    os.makedirs(output_dir, exist_ok=True)
    
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    print(f"\nRendering {num_views} MIP views (radius={radius}, samples={num_samples})...")
    images = []
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        cam_pos = np.array([
            radius * math.cos(angle),
            elevation,
            radius * math.sin(angle)
        ], dtype=np.float32)
        
        with torch.no_grad():
            mip = render_mip_cuda(
                means=params['means'],
                scales=params['scales'],
                intensities=params['intensities'],
                cam_pos=cam_pos,
                target=target,
                width=image_size,
                height=image_size,
                fov_deg=60.0,
                num_samples=num_samples,
                near=0.1,
                far=radius * 2
            )
        
        # Normalize and convert to image
        mip_np = mip.cpu().numpy()
        if mip_np.max() > 0:
            mip_np = mip_np / mip_np.max()
        mip_np = (mip_np * 255).astype(np.uint8)
        
        if i == 0 or i == num_views // 4:
            print(f"  View {i}: intensity range [{mip.min():.4f}, {mip.max():.4f}]")
        
        images.append(mip_np)
        Image.fromarray(mip_np).save(os.path.join(output_dir, f'mip_{i:03d}.png'))
    
    print(f"Saved {num_views} MIP renders to {output_dir}")
    
    # Create GIF
    try:
        frames = [Image.fromarray(img) for img in images]
        frames[0].save(
            os.path.join(output_dir, 'mip_orbit.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f"Created mip_orbit.gif")
    except Exception as e:
        print(f"Could not create GIF: {e}")
    
    return images


def main():
    parser = argparse.ArgumentParser(description='CUDA MIP Renderer for Gaussians')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--volume', type=str, required=True)
    parser.add_argument('--output', type=str, default='renders_mip')
    parser.add_argument('--num_views', type=int, default=36)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--elevation', type=float, default=0.2)
    parser.add_argument('--num_samples', type=int, default=128, help='Samples per ray')
    parser.add_argument('--intensity_threshold', type=float, default=0.15)
    
    args = parser.parse_args()
    
    print(f"Loading volume shape from: {args.volume}")
    volume = tiff.imread(args.volume)
    volume_shape = tuple(volume.shape)
    print(f"  Volume shape: {volume_shape}")
    
    params = load_gaussian_model(args.checkpoint, volume_shape, args.intensity_threshold)
    
    render_orbit_mip(
        params=params,
        num_views=args.num_views,
        image_size=args.image_size,
        output_dir=args.output,
        radius=args.radius,
        elevation=args.elevation,
        num_samples=args.num_samples
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
