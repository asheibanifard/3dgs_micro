"""
Real-time 3D Volume Viewer for Gaussian Reconstruction
Interactive rotation, zoom, and pan with volume rendering

Usage:
    python realtime_volume_viewer.py

Controls:
    - Left mouse drag: Rotate
    - Right mouse drag: Zoom
    - Middle mouse drag: Pan
    - Scroll: Zoom
    - 'r': Reset camera
    - 's': Screenshot
    - 'q': Quit
"""

import torch
import numpy as np
from vedo import Volume, show, Plotter, Text2D

def load_gaussian_volume():
    """Load the trained Gaussian model and render to volume"""
    from models.gaussian_model import GaussianModel
    from utils import get_config, get_data_loader
    
    print("Loading Gaussian model...")
    config = get_config('configs/gaussian_swc.yaml')
    checkpoint = torch.load('outputs/gaussian_swc/tiff/checkpoints/model_iter_15000.pth')
    
    # Create Gaussian model
    gaussians = GaussianModel()
    gaussians._xyz = torch.nn.Parameter(checkpoint['xyz'].cuda())
    gaussians._intensity = torch.nn.Parameter(checkpoint['intensity'].cuda())
    gaussians._scaling = torch.nn.Parameter(checkpoint['scaling'].cuda())
    gaussians._rotation = torch.nn.Parameter(checkpoint['rotation'].cuda())
    gaussians.setup_functions()
    
    # Load target volume for comparison
    config['img_size'] = tuple(config['img_size'])
    data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], 
                                   img_slice=None, train=True, batch_size=1)
    for grid, image in data_loader:
        grid = grid.cuda()
        image = image.cuda()
        break
    
    # Render reconstruction
    print("Rendering Gaussian volume...")
    with torch.no_grad():
        recon = gaussians.grid_sample(grid)
    
    recon_np = recon.squeeze().cpu().numpy()
    target_np = image.squeeze().cpu().numpy()
    
    print(f"Volume shape: {recon_np.shape}")
    print(f"Gaussians: {checkpoint['num_gaussians']}")
    
    return recon_np, target_np, checkpoint['num_gaussians']


def create_interactive_viewer(recon_np, target_np, num_gaussians):
    """Create interactive 3D volume viewer with vedo"""
    
    # Normalize and convert to uint8 for volume rendering
    recon_uint8 = (np.clip(recon_np, 0, 1) * 255).astype(np.uint8)
    target_uint8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
    
    # Create Volume objects
    vol_recon = Volume(recon_uint8)
    vol_target = Volume(target_uint8)
    
    # Set up volume rendering with transfer function
    # This creates a nice visualization for neuron data
    vol_recon.cmap('hot')  # Color map
    vol_recon.alpha([0, 0, 0.1, 0.3, 0.6, 0.9])  # Opacity transfer function
    vol_recon.alphaUnit(1)  # Opacity scaling
    
    vol_target.cmap('gray')
    vol_target.alpha([0, 0, 0.1, 0.3, 0.6, 0.9])
    vol_target.alphaUnit(1)
    
    # Create plotter with two viewports
    plt = Plotter(shape=(1, 2), title='Gaussian Volume Viewer - Drag to Rotate', 
                  size=(1600, 800), bg='black')
    
    # Left: Target volume
    plt.at(0).add(vol_target)
    plt.at(0).add(Text2D('Target Volume', pos='top-center', c='white', s=1.2))
    
    # Right: Gaussian reconstruction
    plt.at(1).add(vol_recon)
    plt.at(1).add(Text2D(f'Gaussian Reconstruction ({num_gaussians} Gaussians)', 
                         pos='top-center', c='white', s=1.2))
    
    # Instructions
    instructions = """
    Controls:
    - Left drag: Rotate
    - Right drag: Zoom  
    - Middle drag: Pan
    - 'r': Reset view
    - 's': Screenshot
    - 'q': Quit
    """
    plt.at(0).add(Text2D(instructions, pos='bottom-left', c='white', s=0.7))
    
    # Link cameras so both volumes rotate together
    plt.at(0).camera.SetParallelProjection(False)
    plt.at(1).camera.SetParallelProjection(False)
    
    print("\n" + "="*60)
    print("INTERACTIVE VOLUME VIEWER")
    print("="*60)
    print("Left: Target | Right: Gaussian Reconstruction")
    print("\nControls:")
    print("  - Left mouse drag: Rotate")
    print("  - Right mouse drag / Scroll: Zoom")
    print("  - Middle mouse drag: Pan")
    print("  - 'r': Reset camera")
    print("  - 's': Save screenshot")
    print("  - 'q': Quit")
    print("="*60 + "\n")
    
    plt.show(interactive=True)


def create_single_volume_viewer(recon_np, num_gaussians):
    """Create single volume viewer with isosurface"""
    from vedo import Volume, Plotter, Text2D
    
    # Normalize
    recon_uint8 = (np.clip(recon_np, 0, 1) * 255).astype(np.uint8)
    
    # Create volume
    vol = Volume(recon_uint8)
    
    # Create isosurfaces at different thresholds using volume method
    iso1 = vol.isosurface(value=30).color('lightblue').alpha(0.3)
    iso2 = vol.isosurface(value=80).color('yellow').alpha(0.5)
    iso3 = vol.isosurface(value=150).color('red').alpha(0.8)
    
    # Create plotter
    plt = Plotter(title=f'Gaussian Neuron ({num_gaussians} Gaussians) - Drag to Rotate',
                  size=(1200, 900), bg='black', bg2='darkblue')
    
    plt.add(iso1, iso2, iso3)
    plt.add(Text2D(f'Gaussian Reconstruction\n{num_gaussians} Gaussians', 
                   pos='top-center', c='white', s=1.2))
    
    instructions = "Left drag: Rotate | Right drag: Zoom | 'q': Quit"
    plt.add(Text2D(instructions, pos='bottom-center', c='white', s=0.8))
    
    print("\n" + "="*60)
    print("ISOSURFACE VIEWER - Drag to rotate!")
    print("="*60)
    
    plt.show(interactive=True)


def create_mip_rotation_viewer(recon_np, num_gaussians):
    """Create viewer showing MIP from different angles"""
    from vedo import Plotter, Picture, Text2D
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate as nd_rotate
    
    # Store current angle
    state = {'angle': 0}
    
    def compute_mip(angle):
        """Compute MIP after rotating volume"""
        if angle == 0:
            rotated = recon_np
        else:
            rotated = nd_rotate(recon_np, angle, axes=(1, 2), reshape=False, order=1)
        return np.max(rotated, axis=0)
    
    # Create initial MIP
    mip = compute_mip(0)
    mip_uint8 = (np.clip(mip, 0, 1) * 255).astype(np.uint8)
    
    # Create picture from MIP
    pic = Picture(mip_uint8)
    
    # Create plotter
    plotter = Plotter(title='MIP Rotation Viewer', size=(1000, 800))
    
    def update_angle(widget, event):
        angle = widget.GetRepresentation().GetValue()
        state['angle'] = angle
        mip = compute_mip(angle)
        mip_uint8 = (np.clip(mip, 0, 1) * 255).astype(np.uint8)
        pic.inputdata().Modified()
        plotter.render()
    
    plotter.add(pic)
    plotter.add(Text2D(f'Gaussian Neuron MIP - {num_gaussians} Gaussians', 
                       pos='top-center', s=1.2))
    
    # Add rotation slider
    plotter.addSlider2D(update_angle, 0, 360, value=0, 
                        title='Rotation Angle', pos=[(0.1, 0.1), (0.9, 0.1)])
    
    plotter.show(interactive=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='volume', 
                        choices=['volume', 'isosurface', 'mip'],
                        help='Rendering mode: volume, isosurface, or mip')
    args = parser.parse_args()
    
    # Load data
    recon_np, target_np, num_gaussians = load_gaussian_volume()
    
    if args.mode == 'volume':
        create_interactive_viewer(recon_np, target_np, num_gaussians)
    elif args.mode == 'isosurface':
        create_single_volume_viewer(recon_np, num_gaussians)
    elif args.mode == 'mip':
        create_mip_rotation_viewer(recon_np, num_gaussians)
