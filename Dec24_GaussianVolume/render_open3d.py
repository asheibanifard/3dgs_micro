#!/usr/bin/env python3
"""
Render Gaussian Volume using Open3D

Visualize the trained Gaussian model as a point cloud with Open3D.
"""

import argparse
import os
import sys
import numpy as np
import torch
import tifffile as tiff

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import open3d as o3d
except ImportError:
    print("Please install open3d: pip install open3d")
    sys.exit(1)

from gaussian_model_cuda import CUDAGaussianModel


def load_gaussian_params(checkpoint_path: str, volume_shape: tuple) -> dict:
    """Load Gaussian model parameters."""
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
    
    # Extract parameters
    D, H, W = volume_shape
    positions = model.positions.detach().cpu().numpy()  # (N, 3) in [0,1]
    scales = model.scales.detach().cpu().numpy()  # (N, 3)
    intensities = model.intensities.detach().cpu().numpy()  # (N,)
    rotations = model.rotations.detach().cpu().numpy()  # (N, 4)
    
    # Scale positions to volume coordinates
    positions_scaled = positions.copy()
    positions_scaled[:, 0] *= D
    positions_scaled[:, 1] *= H
    positions_scaled[:, 2] *= W
    
    print(f"  Num Gaussians: {num_gaussians}")
    print(f"  Positive intensities: {(intensities > 0).sum()}")
    
    return {
        'positions': positions_scaled,
        'positions_normalized': positions,
        'scales': np.abs(scales) + 1e-6,
        'intensities': intensities,
        'rotations': rotations,
        'num_gaussians': num_gaussians,
        'volume_shape': volume_shape
    }


def create_point_cloud(params: dict, intensity_threshold: float = 0.0) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud from Gaussian parameters."""
    positions = params['positions']
    intensities = params['intensities']
    
    # Filter by intensity
    mask = intensities > intensity_threshold
    filtered_positions = positions[mask]
    filtered_intensities = intensities[mask]
    
    print(f"  Filtered {mask.sum()} / {len(mask)} Gaussians (threshold={intensity_threshold})")
    
    # Normalize intensities for coloring
    if len(filtered_intensities) > 0:
        int_norm = (filtered_intensities - filtered_intensities.min()) / (filtered_intensities.max() - filtered_intensities.min() + 1e-8)
    else:
        int_norm = filtered_intensities
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_positions)
    
    # Color by intensity (grayscale or colormap)
    colors = np.zeros((len(filtered_positions), 3))
    colors[:, 0] = int_norm  # Red channel
    colors[:, 1] = int_norm  # Green channel
    colors[:, 2] = int_norm  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def create_gaussian_ellipsoids(
    params: dict, 
    intensity_threshold: float = 0.1,
    max_gaussians: int = 500,
    scale_factor: float = 3.0
) -> list:
    """Create ellipsoid meshes for Gaussians."""
    positions = params['positions']
    scales = params['scales']
    intensities = params['intensities']
    D, H, W = params['volume_shape']
    
    # Filter and sort by intensity
    mask = intensities > intensity_threshold
    indices = np.where(mask)[0]
    
    # Sort by intensity (show brightest)
    sorted_indices = indices[np.argsort(intensities[indices])[::-1]]
    selected = sorted_indices[:max_gaussians]
    
    print(f"  Creating {len(selected)} ellipsoids (top by intensity)")
    
    meshes = []
    
    for idx in selected:
        pos = positions[idx]
        scale = scales[idx] * scale_factor
        intensity = intensities[idx]
        
        # Scale to volume dimensions
        scale_vol = scale * np.array([D, H, W])
        
        # Create sphere and scale to ellipsoid
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        sphere.scale(np.mean(scale_vol), center=sphere.get_center())
        
        # Transform: scale axes differently for ellipsoid
        vertices = np.asarray(sphere.vertices)
        vertices *= scale_vol / np.mean(scale_vol)
        sphere.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Translate to position
        sphere.translate(pos)
        
        # Color by intensity
        int_norm = min(1.0, max(0.0, intensity / (intensities.max() + 1e-8)))
        color = [int_norm, int_norm, int_norm]
        sphere.paint_uniform_color(color)
        
        meshes.append(sphere)
    
    return meshes


def create_camera_frustum(position: np.ndarray, target: np.ndarray, up: np.ndarray,
                          fov: float = 60.0, aspect: float = 1.0, 
                          near: float = 5.0, far: float = 50.0,
                          color: list = [0, 1, 0]) -> o3d.geometry.LineSet:
    """Create a camera frustum visualization."""
    # Camera coordinate system
    z = position - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    
    # Frustum dimensions
    fov_rad = np.radians(fov)
    h_near = 2 * np.tan(fov_rad / 2) * near
    w_near = h_near * aspect
    h_far = 2 * np.tan(fov_rad / 2) * far
    w_far = h_far * aspect
    
    # Near plane corners (in world coordinates)
    center_near = position - z * near
    near_tl = center_near + y * h_near/2 - x * w_near/2
    near_tr = center_near + y * h_near/2 + x * w_near/2
    near_bl = center_near - y * h_near/2 - x * w_near/2
    near_br = center_near - y * h_near/2 + x * w_near/2
    
    # Far plane corners
    center_far = position - z * far
    far_tl = center_far + y * h_far/2 - x * w_far/2
    far_tr = center_far + y * h_far/2 + x * w_far/2
    far_bl = center_far - y * h_far/2 - x * w_far/2
    far_br = center_far - y * h_far/2 + x * w_far/2
    
    # Create line set
    points = [
        position,      # 0: camera position
        near_tl,       # 1
        near_tr,       # 2
        near_bl,       # 3
        near_br,       # 4
        far_tl,        # 5
        far_tr,        # 6
        far_bl,        # 7
        far_br,        # 8
    ]
    
    lines = [
        # From camera to near plane corners
        [0, 1], [0, 2], [0, 3], [0, 4],
        # Near plane
        [1, 2], [2, 4], [4, 3], [3, 1],
        # Far plane
        [5, 6], [6, 8], [8, 7], [7, 5],
        # Connect near to far
        [1, 5], [2, 6], [3, 7], [4, 8],
    ]
    
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return frustum


def create_camera_arrow(position: np.ndarray, target: np.ndarray, 
                        length: float = 30.0, color: list = [0, 1, 0]) -> o3d.geometry.TriangleMesh:
    """Create an arrow showing camera direction."""
    direction = target - position
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Create arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.02,
        cone_radius=length * 0.04,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    
    # Rotate arrow to point in direction
    # Arrow by default points in +Z direction
    z_axis = np.array([0, 0, 1])
    
    # Compute rotation axis and angle
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        
        # Rodrigues rotation
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        arrow.rotate(R, center=[0, 0, 0])
    
    # Translate to position
    arrow.translate(position)
    
    # Color
    arrow.paint_uniform_color(color)
    
    return arrow


def create_orbit_cameras(center: np.ndarray, radius: float, num_cameras: int = 12,
                         volume_shape: tuple = None) -> list:
    """Create camera frustums arranged in orbit around center."""
    geometries = []
    
    D, H, W = volume_shape if volume_shape else (100, 100, 100)
    scale = max(D, H, W)
    
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        
        # Camera position
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * 0.2,  # Slightly above
            radius * np.sin(angle)
        ])
        
        # Color gradient (rainbow)
        hue = i / num_cameras
        color = [
            max(0, min(1, abs(hue * 6 - 3) - 1)),  # R
            max(0, min(1, 2 - abs(hue * 6 - 2))),  # G
            max(0, min(1, 2 - abs(hue * 6 - 4)))   # B
        ]
        
        # Create frustum
        frustum = create_camera_frustum(
            position=cam_pos,
            target=center,
            up=np.array([0, 1, 0]),
            fov=50.0,
            near=scale * 0.05,
            far=scale * 0.3,
            color=color
        )
        geometries.append(frustum)
        
        # Create arrow
        arrow = create_camera_arrow(
            position=cam_pos,
            target=center,
            length=scale * 0.1,
            color=color
        )
        geometries.append(arrow)
        
        # Create camera position sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.02)
        sphere.translate(cam_pos)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    return geometries


def visualize_interactive(params: dict, volume: np.ndarray = None, show_cameras: bool = True):
    """Interactive visualization with Open3D."""
    print("\nCreating visualization...")
    
    # Create point cloud
    pcd = create_point_cloud(params, intensity_threshold=0.05)
    
    # Create ellipsoids for top Gaussians
    ellipsoids = create_gaussian_ellipsoids(params, intensity_threshold=0.1, max_gaussians=200)
    
    # Create coordinate frame
    D, H, W = params['volume_shape']
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(D, H, W) * 0.1, origin=[0, 0, 0]
    )
    
    # Bounding box
    bbox_points = np.array([
        [0, 0, 0], [D, 0, 0], [D, H, 0], [0, H, 0],
        [0, 0, W], [D, 0, W], [D, H, W], [0, H, W]
    ])
    bbox_lines = [[0,1], [1,2], [2,3], [3,0],
                  [4,5], [5,6], [6,7], [7,4],
                  [0,4], [1,5], [2,6], [3,7]]
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(bbox_points)
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(bbox_lines))])
    
    # Create camera visualizations
    camera_geoms = []
    if show_cameras:
        print("  Adding camera frustums and directions...")
        center = np.array([D/2, H/2, W/2])
        radius = max(D, H, W) * 1.5
        camera_geoms = create_orbit_cameras(
            center=center,
            radius=radius,
            num_cameras=12,
            volume_shape=params['volume_shape']
        )
    
    # Visualize
    geometries = [pcd, coord_frame, bbox] + ellipsoids + camera_geoms
    
    print("\nLaunching Open3D viewer...")
    print("  Controls: Mouse to rotate, scroll to zoom, Shift+mouse to pan")
    print("  Press 'Q' to quit")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Gaussian Volume Visualization",
        width=1280,
        height=720,
        point_show_normal=False
    )


def render_to_images(
    params: dict,
    output_dir: str,
    num_views: int = 36,
    image_size: tuple = (512, 512)
):
    """Render views to images using Open3D."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRendering {num_views} views...")
    
    # Create point cloud
    pcd = create_point_cloud(params, intensity_threshold=0.05)
    
    # Create ellipsoids
    ellipsoids = create_gaussian_ellipsoids(params, intensity_threshold=0.1, max_gaussians=300)
    
    # Combine meshes
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in ellipsoids:
        combined_mesh += mesh
    
    # Bounding box for camera positioning
    D, H, W = params['volume_shape']
    center = np.array([D/2, H/2, W/2])
    radius = np.sqrt(D**2 + H**2 + W**2) / 2 * 1.5
    
    # Setup offscreen renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[0], height=image_size[1], visible=False)
    
    # Add geometries
    vis.add_geometry(pcd)
    if len(ellipsoids) > 0:
        vis.add_geometry(combined_mesh)
    
    # Render loop
    images = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * 0.3,
            radius * np.sin(angle)
        ])
        
        # Set camera
        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front((center - cam_pos) / np.linalg.norm(center - cam_pos))
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.5)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Capture
        img = vis.capture_screen_float_buffer(do_render=True)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        images.append(img_np)
        
        # Save
        o3d.io.write_image(os.path.join(output_dir, f'view_{i:03d}.png'), 
                          o3d.geometry.Image(img_np))
    
    vis.destroy_window()
    
    # Create GIF
    try:
        from PIL import Image
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
    
    print(f"Saved {num_views} renders to {output_dir}")
    return images


def main():
    parser = argparse.ArgumentParser(description='Visualize Gaussian Volume with Open3D')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to original volume (for shape)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for rendered images (if provided)')
    parser.add_argument('--num_views', type=int, default=36,
                        help='Number of views for orbit rendering')
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive viewer')
    
    args = parser.parse_args()
    
    # Get volume shape
    print(f"Loading volume from: {args.volume}")
    volume = tiff.imread(args.volume)
    volume_shape = tuple(volume.shape)
    print(f"  Volume shape: {volume_shape}")
    
    # Load model
    params = load_gaussian_params(args.checkpoint, volume_shape)
    
    if args.interactive or args.output is None:
        # Interactive visualization
        visualize_interactive(params, volume)
    
    if args.output:
        # Render to images
        render_to_images(params, args.output, args.num_views)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
