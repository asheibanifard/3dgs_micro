#!/usr/bin/env python
"""
Render MIP projections from 3DGR-CT checkpoint
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse

# Add 3DGR-CT to path
sys.path.insert(0, '/home/armin/Documents/Papers/Publications/2_Publications/paper_3/3dgs_micro/code/3DGR-CT/3DGR-CT')

from gs_utils.general_utils import build_rotation


def load_checkpoint(checkpoint_path):
    """Load 3DGR-CT checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Iteration: {ckpt.get('iteration', 'N/A')}")
    print(f"  Num Gaussians: {ckpt['num_gaussians']}")
    
    config = ckpt.get('config', {})
    print(f"  Volume size: {config.get('img_size', 'N/A')}")
    
    return ckpt


def render_volume_from_gaussians(xyz, scaling, rotation, intensity, volume_shape, device='cuda'):
    """
    Render 3D volume from Gaussian parameters
    
    Args:
        xyz: [N, 3] normalized positions in [0, 1]
        scaling: [N, 3] log-space scales
        rotation: [N, 4] quaternions
        intensity: [N, 1] inverse-sigmoid intensities
        volume_shape: (D, H, W) tuple
    """
    D, H, W = volume_shape
    
    # Convert parameters
    xyz = xyz.to(device)
    scales = torch.exp(scaling).to(device)  # [N, 3]
    intensities = torch.sigmoid(intensity).squeeze(-1).to(device)  # [N]
    
    # Build rotation matrices
    R = build_rotation(rotation.to(device))  # [N, 3, 3]
    
    # Create output volume
    volume = torch.zeros(D, H, W, device=device)
    
    # Convert normalized coords to voxel coords
    xyz_voxel = xyz * torch.tensor([D, H, W], device=device, dtype=torch.float32)
    
    N = xyz.shape[0]
    print(f"Rendering {N} Gaussians to {D}x{H}x{W} volume...")
    
    # Render in chunks for memory efficiency
    chunk_size = 1000
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        
        for i in range(start, end):
            pos = xyz_voxel[i]
            scale = scales[i] * torch.tensor([D, H, W], device=device)  # Scale to voxel space
            opacity = intensities[i].item()
            
            # Compute bounding box (3 sigma)
            sigma_max = scale.max().item() * 3
            
            z0 = max(0, int(pos[0].item() - sigma_max))
            z1 = min(D, int(pos[0].item() + sigma_max) + 1)
            y0 = max(0, int(pos[1].item() - sigma_max))
            y1 = min(H, int(pos[1].item() + sigma_max) + 1)
            x0 = max(0, int(pos[2].item() - sigma_max))
            x1 = min(W, int(pos[2].item() + sigma_max) + 1)
            
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            
            # Create local grid
            zz, yy, xx = torch.meshgrid(
                torch.arange(z0, z1, device=device),
                torch.arange(y0, y1, device=device),
                torch.arange(x0, x1, device=device),
                indexing='ij'
            )
            
            coords = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=1).float()
            diff = coords - pos.unsqueeze(0)  # [M, 3]
            
            # Compute Mahalanobis distance (simplified for axis-aligned)
            dist_sq = (diff ** 2 / (scale ** 2 + 1e-8)).sum(dim=1)
            
            # Gaussian values
            gauss = torch.exp(-0.5 * dist_sq) * opacity
            
            # Add to volume (max for MIP-like effect during rendering)
            local_vol = gauss.reshape(z1-z0, y1-y0, x1-x0)
            volume[z0:z1, y0:y1, x0:x1] = torch.maximum(
                volume[z0:z1, y0:y1, x0:x1], 
                local_vol
            )
        
        if (end - start) == chunk_size:
            print(f"  Processed {end}/{N} Gaussians...")
    
    return volume


def overlay_swc_on_mip(mip_image, swc_path, volume_shape, projection='xy', color=(255, 0, 0), line_width=1):
    """
    Overlay SWC skeleton on a MIP projection image.
    
    Args:
        mip_image: PIL Image (grayscale)
        swc_path: Path to SWC file
        volume_shape: (D, H, W) tuple - the cropped volume dimensions
        projection: 'xy', 'xz', or 'yz'
        color: RGB tuple for skeleton color
        line_width: Line width for skeleton edges
    
    Returns:
        PIL Image (RGB) with skeleton overlay
    """
    from swc_utils import parse_swc_file, swc_to_arrays, get_skeleton_bounds
    
    # Convert grayscale to RGB
    if mip_image.mode == 'L':
        mip_rgb = Image.merge('RGB', (mip_image, mip_image, mip_image))
    else:
        mip_rgb = mip_image.copy()
    
    # Parse SWC
    nodes = parse_swc_file(swc_path)
    positions, radii, parent_ids = swc_to_arrays(nodes)
    
    D, H, W = volume_shape
    
    # Get skeleton bounds - volume is cropped to these bounds
    min_bounds, max_bounds = get_skeleton_bounds(positions)
    extent = max_bounds - min_bounds
    extent = np.where(extent < 1e-6, 1.0, extent)
    
    # SWC file has coordinates (x, y, z) where:
    # - positions[:, 0] = x (maps to W dimension)
    # - positions[:, 1] = y (maps to H dimension)
    # - positions[:, 2] = z (maps to D dimension)
    # 
    # Volume was cropped according to SWC bounding box, so direct mapping:
    # voxel_x = (swc_x - min_x) / extent_x * W
    # voxel_y = (swc_y - min_y) / extent_y * H  
    # voxel_z = (swc_z - min_z) / extent_z * D
    
    # Map SWC coords (x, y, z) to volume voxel coords
    voxel_coords = np.zeros_like(positions)
    voxel_coords[:, 0] = (positions[:, 2] - min_bounds[2]) / extent[2] * (D - 1)  # z -> D
    voxel_coords[:, 1] = (positions[:, 1] - min_bounds[1]) / extent[1] * (H - 1)  # y -> H
    voxel_coords[:, 2] = (positions[:, 0] - min_bounds[0]) / extent[0] * (W - 1)  # x -> W
    
    # Build node ID to index mapping
    node_id_to_idx = {}
    for i, node in enumerate(nodes):
        node_id_to_idx[node.id] = i
    
    # Create drawing context
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mip_rgb)
    
    # Determine which axes to use based on projection
    if projection == 'xy':
        # XY projection: use Y and X (H, W), image is (H, W)
        axis1, axis2 = 1, 2  # Y, X in voxel coords
    elif projection == 'xz':
        # XZ projection: use Z and X (D, W)
        axis1, axis2 = 0, 2
    elif projection == 'yz':
        # YZ projection: use Z and Y (D, H)
        axis1, axis2 = 0, 1
    
    # Draw skeleton edges (lines between connected nodes)
    for i, node in enumerate(nodes):
        if node.parent_id > 0 and node.parent_id in node_id_to_idx:
            parent_idx = node_id_to_idx[node.parent_id]
            
            # Get 2D coordinates for this projection
            x1, y1 = voxel_coords[i, axis2], voxel_coords[i, axis1]
            x2, y2 = voxel_coords[parent_idx, axis2], voxel_coords[parent_idx, axis1]
            
            # Draw line
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
    
    # Draw skeleton nodes as small circles
    for i in range(len(voxel_coords)):
        x, y = voxel_coords[i, axis2], voxel_coords[i, axis1]
        r = max(1, int(radii[i] * 0.5))  # Small radius for visualization
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return mip_rgb


def render_mip_projections(volume, output_dir, swc_path=None, volume_shape=None):
    """Generate MIP projections along each axis"""
    os.makedirs(output_dir, exist_ok=True)
    
    D, H, W = volume.shape
    if volume_shape is None:
        volume_shape = (D, H, W)
    
    # MIP XY (max along Z) - Top view
    mip_xy = volume.max(dim=0)[0]  # [H, W]
    
    # MIP XZ (max along Y) - Front view  
    mip_xz = volume.max(dim=1)[0]  # [D, W]
    
    # MIP YZ (max along X) - Side view
    mip_yz = volume.max(dim=2)[0]  # [D, H]
    
    # Save projections
    def save_mip(mip, name, original_shape):
        mip_np = mip.cpu().numpy()
        # Normalize to [0, 255]
        if mip_np.max() > 0:
            mip_np = (mip_np / mip_np.max() * 255).astype(np.uint8)
        else:
            mip_np = (mip_np * 255).astype(np.uint8)
        
        img = Image.fromarray(mip_np, mode='L')
        path = os.path.join(output_dir, name)
        img.save(path)
        print(f"  Saved {name}: {mip_np.shape} (original volume: {original_shape})")
        return img, path
    
    mip_xy_img, _ = save_mip(mip_xy, 'mip_xy.png', f'{D}x{H}x{W}')
    mip_xz_img, _ = save_mip(mip_xz, 'mip_xz.png', f'{D}x{H}x{W}')
    mip_yz_img, _ = save_mip(mip_yz, 'mip_yz.png', f'{D}x{H}x{W}')
    
    # Overlay SWC skeleton if provided
    if swc_path and os.path.exists(swc_path):
        print(f"\nOverlaying SWC skeleton from: {swc_path}")
        
        # XY overlay
        mip_xy_overlay = overlay_swc_on_mip(mip_xy_img, swc_path, volume_shape, 'xy', 
                                            color=(255, 50, 50), line_width=1)
        overlay_path = os.path.join(output_dir, 'mip_xy_swc_overlay.png')
        mip_xy_overlay.save(overlay_path)
        print(f"  Saved mip_xy_swc_overlay.png: {mip_xy_overlay.size}")
        
        # XZ overlay
        mip_xz_overlay = overlay_swc_on_mip(mip_xz_img, swc_path, volume_shape, 'xz',
                                            color=(255, 50, 50), line_width=1)
        overlay_path = os.path.join(output_dir, 'mip_xz_swc_overlay.png')
        mip_xz_overlay.save(overlay_path)
        print(f"  Saved mip_xz_swc_overlay.png: {mip_xz_overlay.size}")
        
        # YZ overlay
        mip_yz_overlay = overlay_swc_on_mip(mip_yz_img, swc_path, volume_shape, 'yz',
                                            color=(255, 50, 50), line_width=1)
        overlay_path = os.path.join(output_dir, 'mip_yz_swc_overlay.png')
        mip_yz_overlay.save(overlay_path)
        print(f"  Saved mip_yz_swc_overlay.png: {mip_yz_overlay.size}")
    
    print(f"\nMIP projections saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Render MIP from 3DGR-CT checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--volume_size', type=int, nargs=3, default=None, help='Volume size D H W')
    parser.add_argument('--swc_path', type=str, default=None, help='Path to SWC file for overlay')
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    
    # Get volume size from config or args
    if args.volume_size:
        volume_shape = tuple(args.volume_size)
    else:
        config = ckpt.get('config', {})
        img_size = config.get('img_size', [100, 650, 820])
        if isinstance(img_size, int):
            volume_shape = (img_size, img_size, img_size)
        else:
            volume_shape = tuple(img_size)
    
    print(f"Volume shape: {volume_shape}")
    
    # Get SWC path from args or config
    swc_path = args.swc_path
    if swc_path is None:
        config = ckpt.get('config', {})
        swc_path = config.get('swc_path', None)
        if swc_path:
            # Make path relative to checkpoint dir
            ckpt_dir = os.path.dirname(args.checkpoint)
            # Navigate up from checkpoints/model_iter_xxxx.pth to find swc
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_dir)))
            swc_path = os.path.normpath(os.path.join(base_dir, swc_path))
            if not os.path.exists(swc_path):
                # Try relative to 3DGR-CT directory
                swc_path = os.path.normpath(os.path.join(
                    '/home/armin/Documents/Papers/Publications/2_Publications/paper_3/3dgs_micro/code/3DGR-CT',
                    config.get('swc_path', '')
                ))
    
    if swc_path and os.path.exists(swc_path):
        print(f"SWC file: {swc_path}")
    else:
        print(f"SWC file not found: {swc_path}")
        swc_path = None
    
    # Extract Gaussian parameters
    xyz = ckpt['xyz']
    scaling = ckpt['scaling']
    rotation = ckpt['rotation']
    intensity = ckpt['intensity']
    
    print(f"\nGaussian parameters:")
    print(f"  xyz: {xyz.shape}")
    print(f"  scaling: {scaling.shape}")
    print(f"  rotation: {rotation.shape}")
    print(f"  intensity: {intensity.shape}")
    
    # Render volume
    with torch.no_grad():
        volume = render_volume_from_gaussians(
            xyz, scaling, rotation, intensity, 
            volume_shape, device='cuda'
        )
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), 'mip_projections')
    
    # Generate MIP projections with optional SWC overlay
    render_mip_projections(volume, output_dir, swc_path=swc_path, volume_shape=volume_shape)


if __name__ == '__main__':
    main()
