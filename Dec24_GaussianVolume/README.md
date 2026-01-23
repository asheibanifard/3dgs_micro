# Gaussian-Based Volume Data Representation

Implementation based on **[21 Dec. 24]** Algorithm from the research proposal.

## Overview

This module implements a Gaussian-based implicit neural representation for volume data. The volume is represented as a weighted sum of 3D Gaussian basis functions:

$$f(x, y, z) = \sum_{i=1}^{N} w_i \cdot G_i(x, y, z; u_i, \Sigma_i)$$

where each Gaussian basis function is defined as:

$$G_i(x, y, z; u_i, \Sigma_i) = \exp\left\{-\frac{1}{2}(\vec{x} - u_i)^T \Sigma_i^{-1} (\vec{x} - u_i)\right\}$$

## Algorithm Summary

**Input:**
- V: Volume data (voxel values)
- N: Number of Gaussian basis functions
- Initialization parameters for Gaussians (positions $u_i$, covariances $\Sigma_i$, weights $w_i$)

**Output:**
- A trained implicit neural representation of the volume data

**Optimization Problem:**
$$\min_{\{N, u_i, \Sigma_i, w_i\}} \sum_{k=1}^{M} \left\| v_k(x_k, y_k, z_k) - \sum_{i=1}^{N} w_i \cdot G_i(x_k, y_k, z_k; u_i, \Sigma_i) \right\|^2$$

## Files

- `gaussian_model.py` - Gaussian basis functions and volume model
- `losses.py` - Loss functions (MSE + regularization)
- `trainer.py` - Training loop implementation
- `train.py` - Main training script
- `config.yaml` - Default configuration

## Usage

### Basic Training

```bash
python train.py --volume path/to/volume.tif --num_gaussians 5000 --epochs 100
```

### Full Options

```bash
python train.py \
    --volume path/to/volume.tif \
    --num_gaussians 10000 \
    --init_method uniform \
    --epochs 200 \
    --batch_size 8192 \
    --lr 0.01 \
    --optimizer adam \
    --lambda_sparsity 0.001 \
    --lambda_overlap 0.001 \
    --lambda_smoothness 0.001 \
    --output_dir results \
    --device cuda
```

## Algorithm Steps

### Step 1: Initialize Gaussian Basis Functions
- Set initial positions $u_i$ uniformly or based on voxel grid
- Set covariance matrices $\Sigma_i$ to control size and orientation
- Initialize weights $w_i$ randomly or to small uniform values

### Step 2: Construct Implicit Function
Define $f(x, y, z) = \sum_{i=1}^{N} w_i \cdot G_i(x, y, z; u_i, \Sigma_i)$

### Step 3: Define Loss Function
$$L = \frac{1}{M}\sum_{k=1}^{M}(f(x_k, y_k, z_k) - v_k)^2$$

### Step 4: Gradient-Based Optimization
- Use Adam or SGD optimizer
- Learnable parameters: $w_i$, $u_i$, $\Sigma_i$, N

### Step 5: Iterative Training
1. Sample voxel coordinates $(x_k, y_k, z_k)$
2. Forward pass: evaluate $f(\vec{x}_k)$
3. Compute loss
4. Backpropagate gradients
5. Update parameters

### Step 6: Regularization
**Total Loss:**
$$L_{total} = L + L_{sparsity} + L_{overlap} + L_{smoothness}$$

- **Sparsity**: $L_{sparsity} = \lambda_w \sum_{i=1}^{N} |w_i|$
- **Overlap**: $L_{overlap} = \lambda_o \sum_{i \neq j} overlap(G_i, G_j)$
- **Smoothness**: $L_{smoothness} = \lambda_s \sum_{i=1}^{N} \|\nabla_u G_i\|^2$

## Outputs

After training, the following outputs are saved:
- `checkpoints/` - Model checkpoints
- `reconstructed_volume.tif` - Reconstructed volume
- `gaussian_parameters.npz` - Saved Gaussian parameters
- `training_curves.png` - Loss curves
- `reconstruction_comparison.png` - Visual comparison
- `metrics.yaml` - Final metrics (PSNR, MSE, compression ratio)

## API Example

```python
import torch
from Dec24_GaussianVolume import GaussianVolumeModel, create_trainer

# Load volume
volume = torch.randn(100, 256, 256)  # Example volume

# Create model and trainer
model, trainer = create_trainer(
    volume=volume,
    num_gaussians=5000,
    learning_rate=0.01,
    optimizer_type='adam',
    lambda_sparsity=0.001,
    lambda_overlap=0.001,
    lambda_smoothness=0.001,
    device='cuda'
)

# Train
history = trainer.train(num_epochs=100, batch_size=4096)

# Reconstruct volume
reconstructed = model.reconstruct_volume()
```

## References

Based on the research proposal algorithm [21 Dec. 24] for Gaussian-Based Volume Data Representation.
