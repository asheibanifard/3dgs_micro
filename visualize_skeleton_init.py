"""
Visualize SWC Skeleton and Gaussian Initialization

Run this script before training to verify the coordinate mapping is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os

from swc_utils import create_gaussian_params_from_swc


def parse_swc_file(swc_path):
    """Parse SWC file and return nodes."""
    nodes = []
    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                node = {
                    'id': int(parts[0]),
                    'type': int(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'radius': float(parts[5]),
                    'parent': int(parts[6])
                }
                nodes.append(node)
    return nodes


def visualize_skeleton_and_gaussians(swc_path, volume_size, num_gaussians=50000, output_path=None):
    """
    Visualize skeleton and sampled Gaussian positions.
    
    Args:
        swc_path: Path to SWC file
        volume_size: (D, H, W) = (Z, Y, X) volume dimensions
        num_gaussians: Number of Gaussians to sample
        output_path: Path to save the figure (optional)
    """
    print("="*60)
    print("Skeleton and Gaussian Initialization Visualization")
    print("="*60)
    
    # Parse SWC file
    nodes = parse_swc_file(swc_path)
    positions = np.array([[n['x'], n['y'], n['z']] for n in nodes])
    radii = np.array([n['radius'] for n in nodes])
    
    print(f"\nLoaded {len(nodes)} nodes from SWC file: {swc_path}")
    print(f"SWC Position range (X,Y,Z):")
    print(f"  X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
    print(f"  Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
    print(f"  Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    print(f"Radius range: [{radii.min():.3f}, {radii.max():.3f}]")
    
    # Generate Gaussian positions from skeleton
    print(f"\nGenerating {num_gaussians} Gaussians from skeleton...")
    gaussian_params = create_gaussian_params_from_swc(
        swc_path=swc_path,
        num_gaussians=num_gaussians,
        volume_size=volume_size,
        ini_intensity=0.02,
        densify=True,
        points_per_unit=5.0,
        radius_based_density=True
    )
    
    # Extract positions - in (D, H, W) = (Z, Y, X) order
    gaussian_positions = gaussian_params['xyz'].cpu().numpy()
    gaussian_scales = torch.exp(gaussian_params['scaling']).cpu().numpy()
    
    print(f"\nGaussian positions range (normalized [0,1]):")
    print(f"  D (Z): [{gaussian_positions[:,0].min():.4f}, {gaussian_positions[:,0].max():.4f}]")
    print(f"  H (Y): [{gaussian_positions[:,1].min():.4f}, {gaussian_positions[:,1].max():.4f}]")
    print(f"  W (X): [{gaussian_positions[:,2].min():.4f}, {gaussian_positions[:,2].max():.4f}]")
    
    # Denormalize for visualization
    # The normalization uses skeleton bounds with margin=0.05
    swc_min = np.array([positions[:,2].min(), positions[:,1].min(), positions[:,0].min()])  # Z, Y, X
    swc_max = np.array([positions[:,2].max(), positions[:,1].max(), positions[:,0].max()])
    swc_extent = swc_max - swc_min
    
    margin = 0.05
    # gaussian_positions[:, 0] = D = Z
    # gaussian_positions[:, 1] = H = Y  
    # gaussian_positions[:, 2] = W = X
    gauss_z = (gaussian_positions[:, 0] - margin) / (1 - 2*margin) * swc_extent[0] + swc_min[0]
    gauss_y = (gaussian_positions[:, 1] - margin) / (1 - 2*margin) * swc_extent[1] + swc_min[1]
    gauss_x = (gaussian_positions[:, 2] - margin) / (1 - 2*margin) * swc_extent[2] + swc_min[2]
    
    print(f"\nDenormalized Gaussian range:")
    print(f"  X: [{gauss_x.min():.1f}, {gauss_x.max():.1f}]")
    print(f"  Y: [{gauss_y.min():.1f}, {gauss_y.max():.1f}]")
    print(f"  Z: [{gauss_z.min():.1f}, {gauss_z.max():.1f}]")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Build id_to_idx mapping for skeleton connections
    id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    
    # XY projection
    ax = axes[0]
    ax.scatter(gauss_x, gauss_y, c='red', s=0.5, alpha=0.3, label='Gaussians')
    for node in nodes:
        if node['parent'] > 0 and node['parent'] in id_to_idx:
            parent_idx = id_to_idx[node['parent']]
            child_idx = id_to_idx[node['id']]
            ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                    [positions[parent_idx, 1], positions[child_idx, 1]],
                    'b-', linewidth=0.5, alpha=0.5)
    ax.scatter(positions[:,0], positions[:,1], c='blue', s=10, alpha=0.8, label='Skeleton')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'XY Projection ({num_gaussians} Gaussians)')
    ax.legend()
    ax.set_aspect('equal')
    
    # XZ projection
    ax = axes[1]
    ax.scatter(gauss_x, gauss_z, c='red', s=0.5, alpha=0.3, label='Gaussians')
    for node in nodes:
        if node['parent'] > 0 and node['parent'] in id_to_idx:
            parent_idx = id_to_idx[node['parent']]
            child_idx = id_to_idx[node['id']]
            ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                    [positions[parent_idx, 2], positions[child_idx, 2]],
                    'b-', linewidth=0.5, alpha=0.5)
    ax.scatter(positions[:,0], positions[:,2], c='blue', s=10, alpha=0.8, label='Skeleton')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('XZ Projection')
    ax.legend()
    ax.set_aspect('equal')
    
    # YZ projection
    ax = axes[2]
    ax.scatter(gauss_y, gauss_z, c='red', s=0.5, alpha=0.3, label='Gaussians')
    for node in nodes:
        if node['parent'] > 0 and node['parent'] in id_to_idx:
            parent_idx = id_to_idx[node['parent']]
            child_idx = id_to_idx[node['id']]
            ax.plot([positions[parent_idx, 1], positions[child_idx, 1]],
                    [positions[parent_idx, 2], positions[child_idx, 2]],
                    'b-', linewidth=0.5, alpha=0.5)
    ax.scatter(positions[:,1], positions[:,2], c='blue', s=10, alpha=0.8, label='Skeleton')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('YZ Projection')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.suptitle('Skeleton-constrained Gaussian Positions (red) vs Original Skeleton (blue)', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize skeleton and Gaussian initialization')
    parser.add_argument('--swc_path', type=str, 
                        default='/home/armin/Documents/3DGR-CT/10-2900-control-cell-05.oif-C0.v3dpbd.swc',
                        help='Path to SWC file')
    parser.add_argument('--volume_size', type=int, nargs=3, default=[100, 650, 820],
                        help='Volume size (D, H, W) = (Z, Y, X)')
    parser.add_argument('--num_gaussians', type=int, default=50000,
                        help='Number of Gaussians to sample')
    parser.add_argument('--output', type=str, default='skeleton_init_visualization.png',
                        help='Output path for figure')
    
    args = parser.parse_args()
    
    visualize_skeleton_and_gaussians(
        swc_path=args.swc_path,
        volume_size=tuple(args.volume_size),
        num_gaussians=args.num_gaussians,
        output_path=args.output
    )
