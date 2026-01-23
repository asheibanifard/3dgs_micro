"""
CUDA-Accelerated Trainer for Gaussian-Based Volume Data Representation

Uses custom CUDA kernels for fast volume rendering and backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys
from typing import Optional, Dict, List

# Handle both direct run and package import
try:
    from .gaussian_model_cuda import CUDAGaussianModel
except ImportError:
    from gaussian_model_cuda import CUDAGaussianModel


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_val = target.max()
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


class CUDAGaussianTrainer:
    """
    CUDA-Accelerated Trainer for Gaussian Volume Model.
    
    Uses CUDA kernels to render the full volume in one pass,
    then computes loss on the full volume or sampled subset.
    
    Supports adaptive density control (densification):
    - Split: large Gaussians with high gradients → split into smaller ones
    - Clone: small Gaussians with high gradients → duplicate
    - Prune: remove low-intensity or too-large Gaussians
    """
    
    def __init__(
        self,
        model: CUDAGaussianModel,
        volume: torch.Tensor,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam',
        lambda_sparsity: float = 0.001,
        lambda_tv: float = 0.0,
        device: str = 'cuda',
        # Densification parameters
        densify_grad_threshold: float = 0.0002,
        densify_scale_threshold: float = 0.05,
        densify_interval: int = 100,
        densify_start: int = 500,
        densify_stop: int = 15000,
        prune_interval: int = 100,
        max_gaussians: int = 50000
    ):
        """
        Initialize CUDA trainer.
        
        Args:
            model: CUDAGaussianModel to train
            volume: Ground truth volume tensor of shape (D, H, W)
            learning_rate: Learning rate
            optimizer_type: 'adam' or 'sgd'
            lambda_sparsity: Weight for sparsity regularization
            lambda_tv: Weight for total variation regularization
            device: Device to use
            densify_grad_threshold: Gradient threshold for densification
            densify_scale_threshold: Scale threshold for split vs clone
            densify_interval: Densify every N iterations
            densify_start: Start densification after N iterations
            densify_stop: Stop densification after N iterations
            prune_interval: Prune every N iterations
            max_gaussians: Maximum number of Gaussians allowed
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
        # Densification parameters
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_scale_threshold = densify_scale_threshold
        self.densify_interval = densify_interval
        self.densify_start = densify_start
        self.densify_stop = densify_stop
        self.prune_interval = prune_interval
        self.max_gaussians = max_gaussians
        
        # Gradient accumulator for densification
        self.grad_accum = torch.zeros(model.N, device=device)
        self.grad_count = 0
        
        # Normalize and store volume
        self.volume = volume.to(device).float()
        self.volume_max = self.volume.max()
        if self.volume_max > 0:
            self.volume_normalized = self.volume / self.volume_max
        else:
            self.volume_normalized = self.volume
        
        # Optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Loss weights
        self.lambda_sparsity = lambda_sparsity
        self.lambda_tv = lambda_tv
        
        # Training history
        self.history = {
            'total_loss': [],
            'mse_loss': [],
            'sparsity_loss': [],
            'psnr': []
        }
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step using CUDA-accelerated rendering.
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward: render full volume with CUDA kernel
        pred_volume = self.model()  # (D, H, W)
        
        # MSE Loss on full volume
        mse_loss = torch.mean((pred_volume - self.volume_normalized) ** 2)
        
        # Sparsity regularization on intensities
        sparsity_loss = self.lambda_sparsity * torch.mean(torch.abs(self.model.intensities))
        
        # Total loss
        total_loss = mse_loss + sparsity_loss
        
        # Backward (CUDA kernel provides gradients)
        total_loss.backward()
        
        # Accumulate position gradients for densification
        if self.model.positions.grad is not None:
            grad_norm = self.model.positions.grad.norm(dim=1)
            # Handle size mismatch after densification
            if grad_norm.shape[0] == self.grad_accum.shape[0]:
                self.grad_accum += grad_norm
                self.grad_count += 1
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        # Clamp parameters to prevent explosion
        with torch.no_grad():
            # Clamp positions to [0, 1] (normalized coords)
            self.model.positions.data.clamp_(0.0, 1.0)
            # Clamp scales to reasonable range (0.001 to 0.5 in normalized space)
            self.model.scales.data.clamp_(0.001, 0.5)
            # Clamp intensities to prevent extreme values
            self.model.intensities.data.clamp_(-2.0, 2.0)
        
        return {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'sparsity': sparsity_loss.item()
        }
    
    def densification_step(self, iteration: int) -> Dict[str, int]:
        """
        Perform adaptive density control (split/clone/prune).
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Dictionary with densification statistics
        """
        stats = {'split': 0, 'clone': 0, 'prune': 0}
        
        # Check if we should densify
        if iteration < self.densify_start or iteration > self.densify_stop:
            return stats
        
        if iteration % self.densify_interval != 0:
            return stats
        
        # Compute average gradients
        if self.grad_count > 0:
            avg_grads = self.grad_accum / self.grad_count
        else:
            return stats
        
        # Only densify if we're below max Gaussians
        if self.model.N < self.max_gaussians:
            # Split large Gaussians with high gradients
            num_split = self.model.densify_and_split(
                avg_grads,
                self.densify_grad_threshold,
                self.densify_scale_threshold
            )
            stats['split'] = num_split
            
            # Clone small Gaussians with high gradients
            # Recompute grads size after split
            if self.model.N < self.max_gaussians:
                # Need fresh gradients after split - just use threshold without mask
                num_clone = self.model.densify_and_clone(
                    avg_grads[:self.model.N] if avg_grads.shape[0] > self.model.N else 
                    torch.cat([avg_grads, torch.zeros(self.model.N - avg_grads.shape[0], device=self.device)]),
                    self.densify_grad_threshold,
                    self.densify_scale_threshold
                )
                stats['clone'] = num_clone
        
        # Prune low-intensity or too-large Gaussians - DISABLED for now
        # The intensity threshold is too aggressive and causes over-pruning
        # if iteration % self.prune_interval == 0:
        #     num_prune = self.model.prune_gaussians(
        #         intensity_threshold=0.001,  # Very low threshold - only prune nearly zero
        #         scale_threshold=0.6  # Slightly above clamping max (0.5) to avoid over-pruning
        #     )
        #     stats['prune'] = num_prune
        stats['prune'] = 0
        
        # Reset gradient accumulator
        self.grad_accum = torch.zeros(self.model.N, device=self.device)
        self.grad_count = 0
        
        # Rebuild optimizer with new parameters
        if stats['split'] > 0 or stats['clone'] > 0 or stats['prune'] > 0:
            self._rebuild_optimizer()
        
        return stats
    
    def _rebuild_optimizer(self):
        """Rebuild optimizer after densification changes parameter count."""
        old_state = self.optimizer.state_dict()
        
        if isinstance(self.optimizer, optim.Adam):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Note: We lose momentum/Adam state, but this is acceptable for densification

    def train_step_sampled(self, num_samples: int = 50000) -> Dict[str, float]:
        """
        Perform training step with random voxel sampling for larger volumes.
        Uses forward_sampled() which only evaluates CUDA kernel at sampled points.
        This is MUCH faster than full volume evaluation.
        
        Args:
            num_samples: Number of voxels to sample for loss computation
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        D, H, W = self.model.volume_shape
        total_voxels = D * H * W
        
        # Sample random voxel indices
        if num_samples < total_voxels:
            indices = torch.randint(0, total_voxels, (num_samples,), device=self.device)
        else:
            indices = torch.arange(total_voxels, device=self.device)
            num_samples = total_voxels
        
        # Convert linear indices to 3D coordinates (normalized 0-1)
        d_idx = indices // (H * W)
        h_idx = (indices % (H * W)) // W
        w_idx = indices % W
        
        # Normalize to 0-1 range
        sample_points = torch.stack([
            d_idx.float() / (D - 1),
            h_idx.float() / (H - 1),
            w_idx.float() / (W - 1)
        ], dim=-1)  # (num_samples, 3)
        
        # Forward ONLY at sampled points (fast!)
        pred_flat = self.model.forward_sampled(sample_points)
        
        # Get target values
        target_flat = self.volume_normalized.flatten()[indices]
        
        # MSE Loss
        mse_loss = torch.mean((pred_flat - target_flat) ** 2)
        
        # Sparsity
        sparsity_loss = self.lambda_sparsity * torch.mean(torch.abs(self.model.intensities))
        
        total_loss = mse_loss + sparsity_loss
        
        # Backward
        total_loss.backward()
        
        # Accumulate position gradients for densification
        if self.model.positions.grad is not None:
            grad_norm = self.model.positions.grad.norm(dim=1)
            if grad_norm.shape[0] == self.grad_accum.shape[0]:
                self.grad_accum += grad_norm
                self.grad_count += 1
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clamp parameters to prevent explosion
        with torch.no_grad():
            self.model.positions.data.clamp_(0.0, 1.0)
            self.model.scales.data.clamp_(0.001, 0.5)
            self.model.intensities.data.clamp_(-2.0, 2.0)
        
        return {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'sparsity': sparsity_loss.item()
        }
    
    def train(
        self,
        num_epochs: int = 100,
        use_sampling: bool = False,
        num_samples: int = 100000,
        eval_interval: int = 10,
        save_interval: int = 50,
        save_dir: str = 'checkpoints',
        verbose: bool = True,
        use_densification: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop using CUDA acceleration.
        
        Args:
            num_epochs: Number of training epochs
            use_sampling: Use random sampling for loss (faster for large volumes)
            num_samples: Number of samples if using sampling
            eval_interval: Evaluate every N epochs
            save_interval: Save checkpoint every N epochs
            save_dir: Directory for checkpoints
            verbose: Print progress
            use_densification: Enable adaptive density control
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Add densification tracking
        self.history['num_gaussians'] = []
        self.history['densify_split'] = []
        self.history['densify_clone'] = []
        self.history['densify_prune'] = []
        
        pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training")
        
        for epoch in pbar:
            # Train step
            if use_sampling:
                losses = self.train_step_sampled(num_samples)
            else:
                losses = self.train_step()
            
            # Densification
            if use_densification:
                densify_stats = self.densification_step(epoch)
            else:
                densify_stats = {'split': 0, 'clone': 0, 'prune': 0}
            
            # Record history
            self.history['total_loss'].append(losses['total'])
            self.history['mse_loss'].append(losses['mse'])
            self.history['sparsity_loss'].append(losses['sparsity'])
            self.history['num_gaussians'].append(self.model.N)
            self.history['densify_split'].append(densify_stats['split'])
            self.history['densify_clone'].append(densify_stats['clone'])
            self.history['densify_prune'].append(densify_stats['prune'])
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                with torch.no_grad():
                    pred_volume = self.model()
                    # Unnormalize for PSNR
                    if self.volume_max > 0:
                        pred_unnorm = pred_volume * self.volume_max
                    else:
                        pred_unnorm = pred_volume
                    psnr = compute_psnr(pred_unnorm, self.volume)
                    self.history['psnr'].append(psnr)
                
                # Show densification info
                densify_info = ""
                if densify_stats['split'] > 0 or densify_stats['clone'] > 0 or densify_stats['prune'] > 0:
                    densify_info = f" S:{densify_stats['split']} C:{densify_stats['clone']} P:{densify_stats['prune']}"
                
                pbar.set_postfix({
                    'loss': f"{losses['total']:.6f}",
                    'psnr': f"{psnr:.2f}",
                    'N': self.model.N
                })
                
                if verbose and (densify_stats['split'] > 0 or densify_stats['clone'] > 0 or densify_stats['prune'] > 0):
                    tqdm.write(f"  Epoch {epoch}: N={self.model.N}{densify_info}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'),
                    epoch + 1
                )
        
        # Final save
        self.save_checkpoint(os.path.join(save_dir, 'model_final.pth'), num_epochs)
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_params': self.model.get_parameters_dict()
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']
