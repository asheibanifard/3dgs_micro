"""
CUDA-Accelerated Gaussian-Based Volume Data Representation Model

Implementation based on [21 Dec. 24] Algorithm from research proposal.
Uses custom CUDA kernels for fast Gaussian evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path to import CUDA module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gs_utils.Compute_intensity import compute_intensity


class CUDAGaussianModel(nn.Module):
    """
    CUDA-Accelerated Gaussian-Based Volume Data Representation.
    
    Uses custom CUDA kernels for evaluating:
        f(x, y, z) = Σ_{i=1}^{N} w_i * G_i(x, y, z; u_i, Σ_i)
    
    Parameters:
        N: Number of Gaussian basis functions
        u_i: Positions (means) of Gaussians (normalized 0-1)
        Σ_i: Covariance matrices (via scales and rotations)
        w_i: Weights (intensities)
    """
    
    def __init__(
        self,
        num_gaussians: int,
        volume_shape: tuple,
        init_method: str = 'uniform',
        device: str = 'cuda'
    ):
        """
        Initialize CUDA-accelerated Gaussian model.
        
        Args:
            num_gaussians: Number of Gaussian basis functions (N)
            volume_shape: Shape of the volume data (D, H, W)
            init_method: Initialization method ('uniform' or 'grid')
            device: Device to use ('cuda' required)
        """
        super().__init__()
        
        if device != 'cuda':
            raise ValueError("CUDAGaussianModel requires CUDA device")
        
        self.N = num_gaussians
        self.volume_shape = volume_shape
        self.device = device
        self.D, self.H, self.W = volume_shape
        
        # Initialize positions (normalized to 0-1 for CUDA kernel)
        if init_method == 'uniform':
            positions = torch.rand(num_gaussians, 3, device=device)
        elif init_method == 'grid':
            n_per_dim = int(np.ceil(num_gaussians ** (1/3)))
            grid_points = []
            for d in np.linspace(0, 1, n_per_dim):
                for h in np.linspace(0, 1, n_per_dim):
                    for w in np.linspace(0, 1, n_per_dim):
                        grid_points.append([d, h, w])
            positions = torch.tensor(grid_points[:num_gaussians], 
                                     dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        # Learnable parameter: positions (normalized 0-1)
        self.positions = nn.Parameter(positions)  # (N, 3)
        
        # Initialize scales (normalized, controls extent in each dimension)
        # Start with reasonable coverage
        initial_scale = 0.1 / (num_gaussians ** (1/3))
        scales = torch.full((num_gaussians, 3), initial_scale, device=device)
        self.scales = nn.Parameter(scales)  # (N, 3)
        
        # Rotation quaternions for orientation (w, x, y, z)
        rotations = torch.zeros(num_gaussians, 4, device=device)
        rotations[:, 0] = 1.0  # Identity rotation
        self.rotations = nn.Parameter(rotations)  # (N, 4)
        
        # Initialize weights/intensities
        intensities = torch.rand(num_gaussians, device=device) * 0.5
        self.intensities = nn.Parameter(intensities)  # (N,)
        
        # Pre-compute grid points (normalized 0-1)
        self._setup_grid()
    
    def _setup_grid(self):
        """Pre-compute the grid points for CUDA kernel."""
        # Create normalized coordinate grid (0 to 1)
        d_coords = torch.linspace(0, 1, self.D, device=self.device)
        h_coords = torch.linspace(0, 1, self.H, device=self.device)
        w_coords = torch.linspace(0, 1, self.W, device=self.device)
        
        # Create meshgrid (D, H, W)
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Stack to (D*H*W, 3) then reshape to (D, H, W, 3)
        # CUDA kernel expects (1, D, H, W, 3) grid_points
        grid_points = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # (D, H, W, 3)
        self.register_buffer('grid_points', grid_points.reshape(1, self.D, self.H, self.W, 3).contiguous())
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # Normalize quaternions
        quats = quaternions / (torch.norm(quaternions, dim=1, keepdim=True) + 1e-8)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
        
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)
        
        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - x*w)
        
        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    def get_inv_covariances(self) -> torch.Tensor:
        """
        Compute inverse covariance matrices for CUDA kernel.
        
        Σ_i = R_i * diag(s_i^2) * R_i^T
        Σ_i^{-1} = R_i * diag(1/s_i^2) * R_i^T
        
        Returns:
            Inverse covariances flattened to (N, 9)
        """
        # Get scales (ensure positive)
        scales = torch.abs(self.scales) + 1e-6  # (N, 3)
        
        # Get rotation matrices
        R = self._quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)
        
        # Inverse scale matrix (diagonal with 1/s^2)
        inv_S = torch.diag_embed(1.0 / (scales ** 2))  # (N, 3, 3)
        
        # Inverse covariance: Σ^{-1} = R * inv_S * R^T
        inv_cov = torch.bmm(torch.bmm(R, inv_S), R.transpose(1, 2))  # (N, 3, 3)
        
        # Flatten to (N, 9) for CUDA kernel
        return inv_cov.reshape(self.N, 9).contiguous()
    
    def forward(self) -> torch.Tensor:
        """
        Compute volume from Gaussians using CUDA kernel.
        
        Returns:
            Reconstructed volume of shape (D, H, W)
        """
        # Prepare inputs for CUDA kernel
        # gaussian_centers: (N, 3) - normalized positions
        gaussian_centers = self.positions.contiguous()
        
        # grid_points: (1, D, H, W, 3) -> flatten appropriately
        # CUDA kernel expects grid_points as (D*H*W, 3) flattened
        grid_points_flat = self.grid_points.reshape(-1, 3).contiguous()
        
        # intensities: (N,)
        intensities = self.intensities.contiguous()
        
        # inv_covariances: (N, 9)
        inv_covariances = self.get_inv_covariances()
        
        # scalings: (N, 3) - used for bounding box in CUDA kernel
        scalings = (torch.abs(self.scales) + 1e-6).contiguous()
        
        # Output intensity grid: (1, D, H, W)
        intensity_grid = torch.zeros(1, self.D, self.H, self.W, 
                                     device=self.device, dtype=torch.float32)
        
        # Call CUDA kernel
        output = compute_intensity(
            gaussian_centers,
            grid_points_flat,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        # Return volume (D, H, W)
        return output.squeeze(0)
    
    def forward_sampled(self, sample_points: torch.Tensor) -> torch.Tensor:
        """
        Compute intensities at ONLY the given sample points using CUDA kernel.
        Much faster than full forward() for training with sampling.
        
        Args:
            sample_points: (M, 3) tensor of normalized coordinates (0-1)
            
        Returns:
            Intensities at sample points (M,)
        """
        M = sample_points.shape[0]
        
        # Prepare inputs for CUDA kernel
        gaussian_centers = self.positions.contiguous()
        grid_points_flat = sample_points.contiguous()
        intensities = self.intensities.contiguous()
        inv_covariances = self.get_inv_covariances()
        scalings = (torch.abs(self.scales) + 1e-6).contiguous()
        
        # Output intensity grid: CUDA kernel expects (1, D, H, W) but we can reshape
        # We'll create a fake 1D grid: (1, M, 1, 1)
        intensity_grid = torch.zeros(1, M, 1, 1, 
                                     device=self.device, dtype=torch.float32)
        
        # Call CUDA kernel
        output = compute_intensity(
            gaussian_centers,
            grid_points_flat,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        # Return flattened (M,)
        return output.flatten()
    
    def reconstruct_volume(self) -> torch.Tensor:
        """Alias for forward() - reconstruct volume from Gaussians."""
        return self.forward()
    
    @property
    def weights(self):
        """Alias for intensities for compatibility."""
        return self.intensities
    
    @property
    def log_scales(self):
        """Return log of scales for compatibility."""
        return torch.log(torch.abs(self.scales) + 1e-6)
    
    def get_parameters_dict(self) -> dict:
        """Get all learnable parameters as a dictionary."""
        return {
            'positions': self.positions.data.clone(),
            'scales': self.scales.data.clone(),
            'rotations': self.rotations.data.clone(),
            'intensities': self.intensities.data.clone(),
            'num_gaussians': self.N,
            'volume_shape': self.volume_shape
        }
    
    def load_parameters_dict(self, params_dict: dict):
        """Load parameters from a dictionary."""
        self.positions.data = params_dict['positions']
        self.scales.data = params_dict['scales']
        self.rotations.data = params_dict['rotations']
        self.intensities.data = params_dict['intensities']

    def densify_and_split(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float, N_split: int = 2):
        """
        Split Gaussians with high gradients AND large scales.
        
        Args:
            grads: Accumulated position gradients (N,)
            grad_threshold: Threshold for gradient magnitude
            scale_threshold: Minimum scale to trigger split
            N_split: Number of new Gaussians per split
            
        Returns:
            Number of Gaussians split
        """
        # Find Gaussians to split: high gradient AND large scale
        scales_max = torch.abs(self.scales).max(dim=1).values
        mask = (grads > grad_threshold) & (scales_max > scale_threshold)
        
        if mask.sum() == 0:
            return 0
        
        num_split = mask.sum().item()
        
        # Get properties of Gaussians to split
        selected_positions = self.positions[mask]  # (M, 3)
        selected_scales = self.scales[mask]  # (M, 3)
        selected_rotations = self.rotations[mask]  # (M, 4)
        selected_intensities = self.intensities[mask]  # (M,)
        
        # Create new Gaussians by sampling around the original positions
        # New positions: original ± scale * random_offset
        M = selected_positions.shape[0]
        
        # For each split Gaussian, create N_split new ones
        new_positions_list = []
        new_scales_list = []
        new_rotations_list = []
        new_intensities_list = []
        
        for _ in range(N_split):
            # Offset from original position
            offset = torch.randn_like(selected_positions) * torch.abs(selected_scales) * 0.5
            new_positions_list.append(selected_positions + offset)
            
            # Reduce scale for new Gaussians
            new_scales_list.append(selected_scales / (0.8 * N_split))
            
            # Keep same rotation
            new_rotations_list.append(selected_rotations.clone())
            
            # Split intensity among new Gaussians
            new_intensities_list.append(selected_intensities / N_split)
        
        new_positions = torch.cat(new_positions_list, dim=0)
        new_scales = torch.cat(new_scales_list, dim=0)
        new_rotations = torch.cat(new_rotations_list, dim=0)
        new_intensities = torch.cat(new_intensities_list, dim=0)
        
        # Remove original Gaussians and add new ones
        keep_mask = ~mask
        self._update_gaussians(
            positions=torch.cat([self.positions[keep_mask], new_positions], dim=0),
            scales=torch.cat([self.scales[keep_mask], new_scales], dim=0),
            rotations=torch.cat([self.rotations[keep_mask], new_rotations], dim=0),
            intensities=torch.cat([self.intensities[keep_mask], new_intensities], dim=0)
        )
        
        return num_split
    
    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float):
        """
        Clone Gaussians with high gradients AND small scales.
        
        Args:
            grads: Accumulated position gradients (N,)
            grad_threshold: Threshold for gradient magnitude
            scale_threshold: Maximum scale to trigger clone (small Gaussians)
            
        Returns:
            Number of Gaussians cloned
        """
        # Find Gaussians to clone: high gradient AND small scale
        scales_max = torch.abs(self.scales).max(dim=1).values
        mask = (grads > grad_threshold) & (scales_max <= scale_threshold)
        
        if mask.sum() == 0:
            return 0
        
        num_clone = mask.sum().item()
        
        # Clone selected Gaussians with small offset
        new_positions = self.positions[mask] + torch.randn_like(self.positions[mask]) * 0.01
        new_scales = self.scales[mask].clone()
        new_rotations = self.rotations[mask].clone()
        new_intensities = self.intensities[mask].clone()
        
        # Add new Gaussians (keep originals)
        self._update_gaussians(
            positions=torch.cat([self.positions, new_positions], dim=0),
            scales=torch.cat([self.scales, new_scales], dim=0),
            rotations=torch.cat([self.rotations, new_rotations], dim=0),
            intensities=torch.cat([self.intensities, new_intensities], dim=0)
        )
        
        return num_clone
    
    def prune_gaussians(self, intensity_threshold: float = 0.01, scale_threshold: float = 0.5):
        """
        Remove Gaussians with low intensity or excessively large scales.
        
        Args:
            intensity_threshold: Minimum intensity to keep
            scale_threshold: Maximum scale to keep (in normalized space)
            
        Returns:
            Number of Gaussians pruned
        """
        # Keep Gaussians with sufficient intensity and reasonable scale
        scales_max = torch.abs(self.scales).max(dim=1).values
        keep_mask = (self.intensities > intensity_threshold) & (scales_max < scale_threshold)
        
        num_prune = (~keep_mask).sum().item()
        
        if num_prune > 0:
            self._update_gaussians(
                positions=self.positions[keep_mask],
                scales=self.scales[keep_mask],
                rotations=self.rotations[keep_mask],
                intensities=self.intensities[keep_mask]
            )
        
        return num_prune
    
    def _update_gaussians(self, positions, scales, rotations, intensities):
        """Update all Gaussian parameters and count."""
        self.N = positions.shape[0]
        
        # Update parameters (re-create as nn.Parameters)
        self.positions = nn.Parameter(positions.contiguous())
        self.scales = nn.Parameter(scales.contiguous())
        self.rotations = nn.Parameter(rotations.contiguous())
        self.intensities = nn.Parameter(intensities.contiguous())
    
    def reset_opacity(self, value: float = 0.01):
        """Reset all intensities to a low value (called after densification)."""
        # Only reset negative or very low intensities
        with torch.no_grad():
            mask = self.intensities < 0
            self.intensities[mask] = value
