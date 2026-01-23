#!/usr/bin/env python3
"""
Render Volume using Diff-Gaussian-Rasterization

Renders 2D projections of the trained Gaussian model from different viewpoints.
Uses the same conventions as the original 3DGS codebase.
"""

import argparse
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
from PIL import Image

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_gs'))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_model_cuda import CUDAGaussianModel


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """Get world to view transformation matrix."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Get projection matrix."""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    """Minimal camera class compatible with 3DGS rasterizer."""
    
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3, :3]


def create_camera(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int = 512,
    height: int = 512,
    fov_deg: float = 60.0
) -> MiniCam:
    """Create a camera looking at target from position."""
    
    # Compute rotation matrix
    z = position - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    
    R = np.stack([x, y, z], axis=0)  # Camera rotation (3x3)
    t = -R @ position  # Translation
    
    # FOV
    fovy = math.radians(fov_deg)
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    
    znear = 0.01
    zfar = 100.0
    
    # World to view
    world_view = getWorld2View2(R, t)
    world_view_transform = torch.tensor(world_view).transpose(0, 1).cuda().float()
    
    # Projection
    projection = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    
    # Full projection
    full_proj = world_view_transform @ projection
    
    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj
    )


def render_gaussians(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    camera: MiniCam,
    bg_color: torch.Tensor = None
) -> torch.Tensor:
    """Render Gaussians using diff-gaussian-rasterization."""
    
    if bg_color is None:
        bg_color = torch.zeros(3, device='cuda')
    
    # Screen space points
    screenspace_points = torch.zeros_like(means3D, requires_grad=False)
    
    # Rasterization settings
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Render
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    return rendered_image.clamp(0, 1), depth, radii


def load_gaussian_model(checkpoint_path: str, volume_shape: tuple) -> dict:
    """Load Gaussian model and extract parameters."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        num_gaussians = params['num_gaussians']
    else:
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
    
    D, H, W = volume_shape
    positions = model.positions.detach().clone()  # (N, 3) in [0,1]
    
    # Center at origin, scale to reasonable size
    means3D = (positions - 0.5) * 2.0  # Now in [-1, 1]
    
    # Scales - MUST be positive! Use absolute value and scale up
    raw_scales = model.scales.detach()
    scales = torch.abs(raw_scales) + 0.01  # Minimum scale of 0.01
    scales = scales * 0.3  # Scale for visibility
    
    # Rotations (quaternions: w, x, y, z)
    rotations = model.rotations.detach()
    rotations = rotations / (rotations.norm(dim=1, keepdim=True) + 1e-8)
    
    # Intensities -> colors and opacities
    # Filter only positive intensity Gaussians
    intensities = model.intensities.detach()
    
    # Normalize intensities
    intensities_pos = torch.clamp(intensities, 0, None)
    max_int = intensities_pos.max() + 1e-8
    intensities_norm = intensities_pos / max_int
    
    # Grayscale color (white for bright, dark for low intensity)
    colors = intensities_norm.unsqueeze(1).repeat(1, 3)
    
    # Opacities - only show Gaussians with positive intensity
    opacities = torch.where(
        intensities > 0,
        torch.clamp(intensities_norm * 0.8 + 0.2, 0.1, 0.95),
        torch.zeros_like(intensities_norm)
    ).unsqueeze(1)
    
    print(f"  Num Gaussians: {num_gaussians}")
    print(f"  Positive intensity: {(intensities > 0).sum().item()}")
    print(f"  Means3D: [{means3D.min():.3f}, {means3D.max():.3f}]")
    print(f"  Scales (after abs): [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  Colors: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"  Opacities: [{opacities.min():.3f}, {opacities.max():.3f}]")
    
    return {
        'means3D': means3D,
        'scales': scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacities,
        'num_gaussians': num_gaussians
    }


def render_orbit(
    gaussian_params: dict,
    num_views: int = 36,
    image_size: int = 512,
    output_dir: str = 'renders'
):
    """Render orbit around the Gaussians."""
    os.makedirs(output_dir, exist_ok=True)
    
    center = np.array([0.0, 0.0, 0.0])
    radius = 4.0
    
    bg_color = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    
    print(f"Rendering {num_views} views (radius={radius})...")
    images = []
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        cam_pos = np.array([
            radius * math.cos(angle),
            0.5,
            radius * math.sin(angle)
        ])
        
        camera = create_camera(
            position=cam_pos,
            target=center,
            up=np.array([0.0, 1.0, 0.0]),
            width=image_size,
            height=image_size,
            fov_deg=50.0
        )
        
        with torch.no_grad():
            rendered, depth, radii = render_gaussians(
                means3D=gaussian_params['means3D'],
                scales=gaussian_params['scales'],
                rotations=gaussian_params['rotations'],
                opacities=gaussian_params['opacities'],
                colors=gaussian_params['colors'],
                camera=camera,
                bg_color=bg_color
            )
        
        # Check for visible Gaussians
        visible = (radii > 0).sum().item()
        if i == 0:
            print(f"  View 0: {visible} visible Gaussians")
        
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
    
    args = parser.parse_args()
    
    print(f"Loading volume shape from: {args.volume}")
    volume = tiff.imread(args.volume)
    volume_shape = tuple(volume.shape)
    print(f"  Volume shape: {volume_shape}")
    
    gaussian_params = load_gaussian_model(args.checkpoint, volume_shape)
    
    render_orbit(
        gaussian_params=gaussian_params,
        num_views=args.num_views,
        image_size=args.image_size,
        output_dir=args.output
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
