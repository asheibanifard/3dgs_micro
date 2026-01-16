"""
Real-time 3D Gaussian Volume Renderer
Interactive visualization using napari or PyVista
"""

import torch
import numpy as np
import argparse
from models.gaussian_model import GaussianModel
from utils import get_config, get_data_loader


def load_model_and_volume(checkpoint_path, config_path):
    """Load trained Gaussian model and target volume"""
    config = get_config(config_path)
    checkpoint = torch.load(checkpoint_path)
    
    print(f"Loaded {checkpoint['num_gaussians']} Gaussians")
    
    # Create Gaussian model
    gaussians = GaussianModel()
    gaussians._xyz = torch.nn.Parameter(checkpoint['xyz'].cuda())
    gaussians._intensity = torch.nn.Parameter(checkpoint['intensity'].cuda())
    gaussians._scaling = torch.nn.Parameter(checkpoint['scaling'].cuda())
    gaussians._rotation = torch.nn.Parameter(checkpoint['rotation'].cuda())
    gaussians.setup_functions()
    
    # Load target volume
    config['img_size'] = tuple(config['img_size'])
    data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], 
                                   img_slice=None, train=True, batch_size=1)
    for grid, image in data_loader:
        grid = grid.cuda()
        image = image.cuda()
        break
    
    # Render reconstruction
    with torch.no_grad():
        recon = gaussians.grid_sample(grid)
    
    recon_np = recon.squeeze().cpu().numpy()
    target_np = image.squeeze().cpu().numpy()
    
    # Get Gaussian parameters for visualization
    xyz = checkpoint['xyz'].cpu().numpy()
    intensity = torch.sigmoid(checkpoint['intensity']).cpu().numpy().squeeze()
    scales = torch.exp(checkpoint['scaling']).cpu().numpy()
    
    return recon_np, target_np, xyz, intensity, scales, config


def render_with_napari(recon_np, target_np, xyz, intensity, scales, config):
    """Interactive 3D rendering with napari"""
    import napari
    
    D, H, W = recon_np.shape
    xyz_voxel = xyz * np.array([D, H, W])
    
    viewer = napari.Viewer(title='3D Gaussian Neuron Reconstruction')
    
    # Add target volume
    viewer.add_image(target_np, name='Target Volume', 
                     colormap='gray', opacity=0.7,
                     contrast_limits=[0, 1])
    
    # Add reconstruction volume
    viewer.add_image(recon_np, name='Gaussian Reconstruction', 
                     colormap='magma', opacity=0.7,
                     contrast_limits=[0, 1], visible=False)
    
    # Add Gaussian centers as points
    # napari expects (z, y, x) order
    points = xyz_voxel[:, [0, 1, 2]]  # D, H, W
    
    viewer.add_points(points, 
                      name='Gaussian Centers',
                      size=scales.mean(axis=1) * 50,
                      face_color=intensity,
                      face_colormap='hot',
                      opacity=0.8,
                      visible=False)
    
    print("\n" + "="*60)
    print("NAPARI CONTROLS:")
    print("="*60)
    print("  - Scroll wheel: Zoom in/out")
    print("  - Left drag: Rotate view")
    print("  - Shift + drag: Pan")
    print("  - Toggle layers in left panel")
    print("  - Use slider to scroll through Z slices")
    print("  - Press '3' for 3D view, '2' for 2D slice view")
    print("="*60)
    
    napari.run()


def render_with_pyvista(recon_np, target_np, xyz, intensity, scales, config):
    """Interactive 3D rendering with PyVista"""
    import pyvista as pv
    
    D, H, W = recon_np.shape
    xyz_voxel = xyz * np.array([D, H, W])
    
    # Create plotter
    plotter = pv.Plotter(title='3D Gaussian Neuron Reconstruction')
    
    # Create volume from reconstruction
    grid = pv.ImageData(dimensions=(W, H, D))
    grid.point_data['values'] = recon_np.T.flatten(order='F')
    
    # Add volume rendering
    plotter.add_volume(grid, cmap='gray', opacity='sigmoid', 
                       scalar_bar_args={'title': 'Intensity'})
    
    # Add Gaussian centers as spheres
    point_cloud = pv.PolyData(xyz_voxel[:, [2, 1, 0]])  # PyVista uses x, y, z
    point_cloud['intensity'] = intensity
    point_cloud['size'] = scales.mean(axis=1)
    
    # Add points (optional, can be toggled)
    # plotter.add_mesh(point_cloud, scalars='intensity', cmap='hot', 
    #                  point_size=5, render_points_as_spheres=True)
    
    plotter.show_axes()
    plotter.add_text(f'{len(xyz)} Gaussians', position='upper_left')
    
    print("\n" + "="*60)
    print("PYVISTA CONTROLS:")
    print("="*60)
    print("  - Left drag: Rotate view")
    print("  - Right drag: Zoom")
    print("  - Middle drag: Pan")
    print("  - 'r': Reset camera")
    print("  - 'q': Quit")
    print("="*60)
    
    plotter.show()


def render_with_plotly(recon_np, target_np, xyz, intensity, scales, config):
    """Interactive 3D rendering with Plotly (browser-based)"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    D, H, W = recon_np.shape
    xyz_voxel = xyz * np.array([D, H, W])
    
    # Create isosurface from reconstruction
    X, Y, Z = np.mgrid[0:D, 0:H, 0:W]
    
    fig = go.Figure()
    
    # Add isosurface of reconstruction
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=recon_np.flatten(),
        isomin=0.15,
        isomax=0.8,
        opacity=0.3,
        surface_count=3,
        colorscale='gray',
        name='Reconstruction'
    ))
    
    # Add Gaussian centers as scatter
    fig.add_trace(go.Scatter3d(
        x=xyz_voxel[:, 0],
        y=xyz_voxel[:, 1],
        z=xyz_voxel[:, 2],
        mode='markers',
        marker=dict(
            size=scales.mean(axis=1) * 20,
            color=intensity,
            colorscale='Hot',
            opacity=0.6
        ),
        name='Gaussians',
        visible='legendonly'
    ))
    
    fig.update_layout(
        title=f'3D Gaussian Neuron ({len(xyz)} Gaussians)',
        scene=dict(
            xaxis_title='D (Z)',
            yaxis_title='H (Y)',
            zaxis_title='W (X)',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )
    
    # Save to HTML for interactive viewing
    fig.write_html('outputs/gaussian_swc/tiff/interactive_3d.html')
    print("Saved interactive viewer to: outputs/gaussian_swc/tiff/interactive_3d.html")
    print("Open in browser for interactive 3D viewing!")
    
    fig.show()


def render_mip_animation(recon_np, target_np, output_path='outputs/gaussian_swc/tiff/mip_rotation.gif'):
    """Create rotating MIP animation"""
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate
    import imageio
    
    print("Creating MIP rotation animation...")
    
    frames = []
    for angle in range(0, 360, 5):
        # Rotate volume
        rotated = rotate(recon_np, angle, axes=(1, 2), reshape=False, order=1)
        
        # MIP along rotation axis
        mip = np.max(rotated, axis=0)
        
        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mip, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'MIP Rotation: {angle}°')
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=15)
    print(f"Saved animation to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Real-time 3D Gaussian Volume Renderer')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/gaussian_swc/tiff/checkpoints/model_iter_15000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                        default='configs/gaussian_swc.yaml',
                        help='Path to config file')
    parser.add_argument('--renderer', type=str, default='napari',
                        choices=['napari', 'pyvista', 'plotly', 'gif'],
                        help='Rendering backend to use')
    args = parser.parse_args()
    
    print("Loading model and volume...")
    recon_np, target_np, xyz, intensity, scales, config = load_model_and_volume(
        args.checkpoint, args.config
    )
    
    print(f"Volume shape: {recon_np.shape}")
    print(f"Gaussians: {len(xyz)}")
    
    if args.renderer == 'napari':
        render_with_napari(recon_np, target_np, xyz, intensity, scales, config)
    elif args.renderer == 'pyvista':
        render_with_pyvista(recon_np, target_np, xyz, intensity, scales, config)
    elif args.renderer == 'plotly':
        render_with_plotly(recon_np, target_np, xyz, intensity, scales, config)
    elif args.renderer == 'gif':
        render_mip_animation(recon_np, target_np)


if __name__ == '__main__':
    main()
