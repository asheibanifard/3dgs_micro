# 3DGR-CT: Sparse-View CT Reconstruction with a 3D Gaussian Representation


## Gaussian Initialization Methods

This implementation supports two methods for initializing Gaussian positions:

### 1. FBP Image-Based Initialization (Original)
- Uses Filtered Back Projection (FBP) reconstruction
- Gradient-based density allocation
- May place Gaussians in void regions

### 2. SWC Skeleton-Based Initialization (New)
- Uses SWC neuron skeleton files for positioning
- **Gaussians placed ONLY along neuron structure**
- **Skeleton density + radius** for allocation
- **Avoids void regions** - more efficient for sparse structures

| Feature | FBP-Based | SWC-Based |
|---------|-----------|-----------|
| Position Source | FBP image gradients | SWC skeleton geometry |
| Density Allocation | Gradient-based | Skeleton radius-based |
| Void Region Handling | May include empty space | Avoids void regions |
| Best For | Dense structures | Sparse neuron structures |


# Instructions for Running Code

### Standard FBP-based initialization:
```
python train_ct_recon.py --config configs/gaussian.yaml
```

### SWC skeleton-based initialization:
```
python train_ct_recon.py --config configs/gaussian_swc.yaml
```

Or with command line arguments:
```
python train_ct_recon.py --config configs/gaussian.yaml --use_swc --swc_path path/to/skeleton.swc
```

## SWC Configuration Options

In the config file, you can set:
```yaml
use_swc: true                    # Enable SWC-based initialization
swc_path: path/to/skeleton.swc   # Path to SWC file
swc_densify: true                # Interpolate skeleton points
swc_points_per_unit: 5.0         # Interpolation density
swc_radius_density: true         # More Gaussians for thicker regions
```

## 🤝Acknowledgement

Our repo is built upon [Gasussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Splat image](https://github.com/szymanowiczs/splatter-image), [NeRP](https://github.com/liyues/NeRP) and [ODL](https://github.com/odlgroup/odl). Thanks to their work.
