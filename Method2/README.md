# 3DGS Microscopy - Method 2 (VolSplat-based)

Voxel-based 3D Gaussian Splatting for Volumetric Microscopy Reconstruction

## Overview

Method 2 implements a **voxel-grid-based approach** for fitting 3D Gaussians to microscopy volumes, inspired by the [VolSplat](https://github.com/autonomousvision/volsplat) framework. Unlike Method 1 (skeleton-based), this approach:

- Initializes **one Gaussian per occupied voxel** above an intensity threshold
- Uses **additive blending** (fluorescence model) instead of alpha compositing
- Supports **adaptive densification** (clone/split/prune) during optimization
- Optimizes via **L1 + SSIM reconstruction loss**

## Features

- **Voxel-aligned initialization** - Gaussians placed at voxel centers
- **Adaptive density control** - Clone, split, and prune Gaussians dynamically
- **Multi-scale losses** - L1, SSIM, and optional LPIPS
- **Configurable via YAML** - All parameters in `config.yml`
- **Block-wise processing** - Handle large volumes by splitting into blocks

## Installation

```bash
# Clone repository
git clone https://github.com/asheibanifard/3dgs_microscopy.git
cd 3dgs_microscopy/Method2

# Install dependencies
pip install torch torchvision numpy scipy tifffile matplotlib tqdm pyyaml

# Optional: CUDA Gaussian rasterization
pip install diff-gaussian-rasterization  # From 3DGS repo
```

## Project Structure

```
Method2/
├── README.md                 # This file
├── config.yml               # Main configuration file
├── config_loader.py         # YAML config parser
├── optimize_gaussians.py    # Main optimization script
├── loss.py                  # Loss functions (L1, SSIM, LPIPS)
├── densification.py         # Adaptive density control
├── voxel_to_gaussians.py    # Voxel-based initialization
├── data.py                  # Data loading utilities
├── EDA.ipynb               # Exploratory data analysis notebook
├── VolSplat/               # Reference VolSplat implementation
├── output/                 # Output directory
└── *.tif, *.swc            # Example data files
```

## Usage

### Basic Training

```python
from optimize_gaussians import GaussianModel, optimize_gaussians
from data import load_volume

# Load volume
volume = load_volume("10-2900-control-cell-05_cropped_corrected.tif")

# Initialize and optimize
model, losses = optimize_gaussians(
    volume=volume,
    config_path="config.yml",
    output_dir="output/"
)
```

### Command Line

```bash
# Run optimization with default config
python optimize_gaussians.py

# Run with custom config
python optimize_gaussians.py --config my_config.yml
```

### Block-wise Processing

For large volumes, process as blocks:

```python
import numpy as np

# Load pre-extracted block
block = np.load("block_133.npy")

# Process block
model, losses = optimize_gaussians(volume=block, config_path="config.yml")
```

## Configuration

Key parameters in `config.yml`:

```yaml
# Data settings
data:
  intensity_threshold: 0.1    # Voxel selection threshold
  voxel_spacing: [1.0, 1.0, 1.0]

# Model settings
model:
  scale_init: 0.5            # Initial Gaussian scale
  scale_min: 0.001
  scale_max: 10.0

# Optimization
optimization:
  num_iterations: 10000
  lr_means: 0.001            # Position learning rate
  lr_scales: 0.005           # Scale learning rate
  lr_opacities: 0.05
  lr_intensities: 0.05

# Losses
loss:
  lambda_l1: 0.8
  lambda_ssim: 0.2
  lambda_opacity_reg: 0.01

# Densification
densification:
  enabled: true
  interval: 100              # Densify every N iterations
  start_iter: 500
  until_iter: 15000
  grad_threshold: 0.0002
  opacity_threshold: 0.005
```

## Algorithm

### 1. Initialization
```
For each voxel v with intensity I(v) > threshold:
    Create Gaussian G with:
        - position μ = voxel center
        - scale σ = voxel_size * scale_init
        - rotation q = identity
        - opacity α = sigmoid⁻¹(I(v))
        - intensity = I(v)
```

### 2. Rendering (Additive)
```
For each query point x:
    I(x) = Σᵢ αᵢ * Iᵢ * exp(-0.5 * (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ))
```

### 3. Loss Function
```
L = λ₁ * L1(I_pred, I_target) + λ₂ * (1 - SSIM(I_pred, I_target))
  + λ_reg * (opacity_reg + scale_reg)
```

### 4. Densification
- **Clone**: Small Gaussians with high gradients → duplicate at same position
- **Split**: Large Gaussians with high gradients → split into multiple smaller ones
- **Prune**: Remove low-opacity or oversized Gaussians

## Loss Functions

| Loss | Description |
|------|-------------|
| `l1_loss` | Per-voxel L1 reconstruction |
| `ssim_loss` | Structural similarity (2D slice-wise or 3D) |
| `opacity_reg` | Encourage sparse opacity distribution |
| `scale_reg` | Penalize overly large Gaussians |
| `lpips_loss` | Optional perceptual loss (requires lpips) |

## Densification Strategy

```
Every 100 iterations from iter 500 to 15000:
    1. Compute gradient magnitude for each Gaussian
    2. Clone: grad > threshold AND scale < clone_threshold
    3. Split: grad > threshold AND scale > split_threshold
    4. Prune: opacity < 0.005 OR scale > max_scale

Every 3000 iterations:
    Reset all opacities to 0.01 (prevents opacity collapse)
```

## Output

The optimization produces:
- `gaussians.pth` - Trained Gaussian parameters
- `losses.json` - Training loss history
- `rendered_*.png` - MIP projections at checkpoints
- `config_used.yml` - Configuration snapshot

## Comparison with Method 1

| Aspect | Method 1 (Skeleton) | Method 2 (VolSplat) |
|--------|--------------------|--------------------|
| Initialization | SWC skeleton points | Voxel grid |
| Blending | Alpha (front-to-back) | Additive |
| Structure prior | Skeleton constraint | None |
| Best for | Neurons with SWC | General volumes |
| Density control | Skeleton-guided | Gradient-based |

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., 2023
- [VolSplat](https://github.com/autonomousvision/volsplat) - Volumetric Gaussian Splatting

## Citation

```bibtex
@article{3dgs_microscopy_method2,
  title={Voxel-based 3D Gaussian Splatting for Volumetric Microscopy},
  author={Sheibanifard, A.},
  year={2026}
}
```

## License

MIT License
