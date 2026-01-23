#!/usr/bin/env python3
"""
CUDA-Accelerated Training Script for Gaussian-Based Volume Data Representation

Uses custom CUDA kernels for fast Gaussian evaluation.
Based on [21 Dec. 24] Algorithm.

Usage:
    python train_cuda.py --volume <volume.tif> --num_gaussians 5000 --epochs 100
"""

import argparse
import os
import sys
import torch
import numpy as np
import tifffile as tiff
from datetime import datetime

from gaussian_model_cuda import CUDAGaussianModel
from trainer_cuda import CUDAGaussianTrainer


def load_volume(path: str) -> torch.Tensor:
    """Load volume from TIFF file."""
    print(f"Loading volume from: {path}")
    volume = tiff.imread(path)
    volume = torch.tensor(volume, dtype=torch.float32)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    return volume


def save_results(
    model: CUDAGaussianModel,
    trainer: CUDAGaussianTrainer,
    output_dir: str,
    volume: torch.Tensor
):
    """Save training results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reconstructed volume
    with torch.no_grad():
        recon = model.reconstruct_volume()
        # Unnormalize
        if trainer.volume_max > 0:
            recon = recon * trainer.volume_max
        recon_np = recon.cpu().numpy().astype(np.float32)
        tiff.imwrite(os.path.join(output_dir, 'reconstructed.tif'), recon_np)
    
    # Save training history
    import json
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    # Save model parameters
    params = model.get_parameters_dict()
    params_save = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                   for k, v in params.items()}
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params_save, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='CUDA-Accelerated Gaussian Volume Training'
    )
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to volume TIFF file')
    parser.add_argument('--num_gaussians', type=int, default=5000,
                        help='Number of Gaussian basis functions')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lambda_sparsity', type=float, default=0.001,
                        help='Sparsity regularization weight')
    parser.add_argument('--init_method', type=str, default='uniform',
                        choices=['uniform', 'grid'],
                        help='Initialization method')
    parser.add_argument('--use_sampling', action='store_true',
                        help='Use random sampling for loss (faster)')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples when using sampling')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluate every N epochs')
    
    # Densification arguments
    parser.add_argument('--densify', action='store_true',
                        help='Enable adaptive density control (split/clone/prune)')
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002,
                        help='Gradient threshold for densification')
    parser.add_argument('--densify_scale_threshold', type=float, default=0.05,
                        help='Scale threshold for split vs clone')
    parser.add_argument('--densify_interval', type=int, default=100,
                        help='Densify every N iterations')
    parser.add_argument('--densify_start', type=int, default=500,
                        help='Start densification after N iterations')
    parser.add_argument('--densify_stop', type=int, default=15000,
                        help='Stop densification after N iterations')
    parser.add_argument('--max_gaussians', type=int, default=50000,
                        help='Maximum number of Gaussians')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this script!")
        sys.exit(1)
    
    print("=" * 60)
    print("CUDA-Accelerated Gaussian-Based Volume Data Representation")
    print("Implementation based on [21 Dec. 24] Algorithm")
    print("=" * 60)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Load volume
    volume = load_volume(args.volume)
    volume_shape = tuple(volume.shape)
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/cuda_gaussian_{timestamp}"
    
    print(f"\nConfiguration:")
    print(f"  Number of Gaussians: {args.num_gaussians}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Sparsity weight: {args.lambda_sparsity}")
    print(f"  Use sampling: {args.use_sampling}")
    if args.use_sampling:
        print(f"  Num samples: {args.num_samples}")
    print(f"  Densification: {args.densify}")
    if args.densify:
        print(f"    Grad threshold: {args.densify_grad_threshold}")
        print(f"    Scale threshold: {args.densify_scale_threshold}")
        print(f"    Interval: {args.densify_interval}")
        print(f"    Start/Stop: {args.densify_start}/{args.densify_stop}")
        print(f"    Max Gaussians: {args.max_gaussians}")
    print(f"  Output: {args.output_dir}")
    
    # Create model
    print("\nInitializing model...")
    model = CUDAGaussianModel(
        num_gaussians=args.num_gaussians,
        volume_shape=volume_shape,
        init_method=args.init_method,
        device='cuda'
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = CUDAGaussianTrainer(
        model=model,
        volume=volume,
        learning_rate=args.lr,
        optimizer_type=args.optimizer,
        lambda_sparsity=args.lambda_sparsity,
        device='cuda',
        densify_grad_threshold=args.densify_grad_threshold,
        densify_scale_threshold=args.densify_scale_threshold,
        densify_interval=args.densify_interval,
        densify_start=args.densify_start,
        densify_stop=args.densify_stop,
        max_gaussians=args.max_gaussians
    )
    
    # Train
    print("\nStarting CUDA-accelerated training...")
    history = trainer.train(
        num_epochs=args.epochs,
        use_sampling=args.use_sampling,
        num_samples=args.num_samples,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
        verbose=True,
        use_densification=args.densify
    )
    
    # Save results
    save_results(model, trainer, args.output_dir, volume)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    if history['psnr']:
        print(f"  Final PSNR: {history['psnr'][-1]:.2f} dB")
    print(f"  Final Loss: {history['total_loss'][-1]:.6f}")
    print(f"  Final Gaussians: {model.N}")
    if args.densify and 'num_gaussians' in history:
        print(f"  Gaussians: {args.num_gaussians} → {model.N}")
        total_split = sum(history.get('densify_split', []))
        total_clone = sum(history.get('densify_clone', []))
        total_prune = sum(history.get('densify_prune', []))
        print(f"  Densification: Split={total_split}, Clone={total_clone}, Prune={total_prune}")


if __name__ == '__main__':
    main()
