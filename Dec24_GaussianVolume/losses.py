"""
Loss Functions for Gaussian-Based Volume Data Representation

Implementation based on [21 Dec. 24] Algorithm from research proposal.

Loss Functions:
1. MSE Loss (Main reconstruction loss)
2. Sparsity Regularization (L1 on weights)
3. Overlap Regularization (penalize Gaussian overlap)
4. Smoothness Regularization (smooth Gaussian parameters)
5. Total Loss (combination of all losses)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Mean Squared Error Loss for volume reconstruction.
    
    L = (1/M) * Σ_{k=1}^{M} (f(x_k, y_k, z_k) - v_k)^2
    
    where M is the number of ground truth voxels,
    (x_k, y_k, z_k) are voxel coordinates,
    v_k are the corresponding ground truth values.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and ground truth values.
        
        Args:
            predicted: Predicted voxel values of shape (M,)
            ground_truth: Ground truth voxel values of shape (M,)
            
        Returns:
            MSE loss value
        """
        return self.mse(predicted, ground_truth)


class SparsityRegularization(nn.Module):
    """
    Weight Sparsity Regularization.
    
    L_sparsity = λ_w * Σ_{i=1}^{N} |w_i|
    
    Encourages sparsity in the weights w_i.
    """
    
    def __init__(self, lambda_w: float = 0.01):
        """
        Args:
            lambda_w: Regularization hyperparameter
        """
        super().__init__()
        self.lambda_w = lambda_w
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity regularization.
        
        Args:
            weights: Gaussian weights of shape (N,)
            
        Returns:
            Sparsity loss value
        """
        return self.lambda_w * torch.sum(torch.abs(weights))


class OverlapRegularization(nn.Module):
    """
    Overlap Regularization.
    
    L_overlap = λ_o * Σ_{i≠j} overlap(G_i, G_j)
    
    Penalizes excessive overlap between Gaussians to encourage
    distinct representations.
    
    The overlap between two Gaussians is computed using the
    Bhattacharyya coefficient (integral of sqrt(G_i * G_j)).
    """
    
    def __init__(self, lambda_o: float = 0.01):
        """
        Args:
            lambda_o: Regularization hyperparameter
        """
        super().__init__()
        self.lambda_o = lambda_o
    
    def forward(
        self,
        positions: torch.Tensor,
        covariance: torch.Tensor,
        max_pairs: int = 1000
    ) -> torch.Tensor:
        """
        Compute overlap regularization.
        
        For computational efficiency, we use an approximation based on
        the distance between Gaussian centers relative to their sizes.
        
        Args:
            positions: Gaussian positions of shape (N, 3)
            covariance: Covariance matrices of shape (N, 3, 3)
            max_pairs: Maximum number of pairs to consider for efficiency
            
        Returns:
            Overlap loss value
        """
        N = positions.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Compute pairwise distances between Gaussian centers
        # diff: (N, N, 3)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        distances = torch.norm(diff, dim=2)  # (N, N)
        
        # Compute average scales (approximate radius) for each Gaussian
        # Use trace of covariance as measure of size
        scales = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2).mean(dim=1))  # (N,)
        
        # Combined scale for each pair
        combined_scales = scales.unsqueeze(1) + scales.unsqueeze(0)  # (N, N)
        
        # Overlap measure: exp(-distance^2 / (2 * combined_scale^2))
        # Higher value means more overlap
        overlap_measure = torch.exp(-distances**2 / (2 * combined_scales**2 + 1e-6))
        
        # Mask diagonal (self-overlap)
        mask = ~torch.eye(N, dtype=torch.bool, device=positions.device)
        overlap_values = overlap_measure[mask]
        
        # Sum overlap (divide by 2 since symmetric)
        total_overlap = torch.sum(overlap_values) / 2.0
        
        return self.lambda_o * total_overlap


class SmoothnessRegularization(nn.Module):
    """
    Smoothness Regularization.
    
    L_smoothness = λ_s * Σ_{i=1}^{N} ||∇_u G_i||^2
    
    Enforces smooth transitions in the Gaussian parameters.
    
    We approximate this by penalizing large variations in
    neighboring Gaussian parameters.
    """
    
    def __init__(self, lambda_s: float = 0.01, num_neighbors: int = 5):
        """
        Args:
            lambda_s: Regularization hyperparameter
            num_neighbors: Number of nearest neighbors to consider
        """
        super().__init__()
        self.lambda_s = lambda_s
        self.num_neighbors = num_neighbors
    
    def forward(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        log_scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness regularization.
        
        Args:
            positions: Gaussian positions of shape (N, 3)
            weights: Gaussian weights of shape (N,)
            log_scales: Log scales of shape (N, 3)
            
        Returns:
            Smoothness loss value
        """
        N = positions.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Compute pairwise distances
        distances = torch.cdist(positions, positions)  # (N, N)
        
        # Get k nearest neighbors for each Gaussian
        k = min(self.num_neighbors, N - 1)
        _, indices = torch.topk(distances, k + 1, largest=False, dim=1)
        # Exclude self (first index)
        neighbor_indices = indices[:, 1:]  # (N, k)
        
        # Compute smoothness for weights
        weight_diff = weights.unsqueeze(1) - weights[neighbor_indices]  # (N, k)
        weight_smoothness = torch.mean(weight_diff ** 2)
        
        # Compute smoothness for scales
        scale_diff = log_scales.unsqueeze(1) - log_scales[neighbor_indices]  # (N, k, 3)
        scale_smoothness = torch.mean(scale_diff ** 2)
        
        return self.lambda_s * (weight_smoothness + scale_smoothness)


class TotalLoss(nn.Module):
    """
    Total Loss combining all loss terms.
    
    L_total = L + L_sparsity + L_overlap + L_smoothness
    
    where L is the main reconstruction loss (MSE).
    """
    
    def __init__(
        self,
        lambda_sparsity: float = 0.01,
        lambda_overlap: float = 0.01,
        lambda_smoothness: float = 0.01,
        use_sparsity: bool = True,
        use_overlap: bool = True,
        use_smoothness: bool = True
    ):
        """
        Args:
            lambda_sparsity: Weight for sparsity regularization
            lambda_overlap: Weight for overlap regularization
            lambda_smoothness: Weight for smoothness regularization
            use_sparsity: Whether to use sparsity regularization
            use_overlap: Whether to use overlap regularization
            use_smoothness: Whether to use smoothness regularization
        """
        super().__init__()
        
        self.reconstruction_loss = ReconstructionLoss()
        
        self.use_sparsity = use_sparsity
        self.use_overlap = use_overlap
        self.use_smoothness = use_smoothness
        
        if use_sparsity:
            self.sparsity_loss = SparsityRegularization(lambda_sparsity)
        if use_overlap:
            self.overlap_loss = OverlapRegularization(lambda_overlap)
        if use_smoothness:
            self.smoothness_loss = SmoothnessRegularization(lambda_smoothness)
    
    def forward(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor,
        model=None
    ) -> dict:
        """
        Compute total loss.
        
        Args:
            predicted: Predicted voxel values of shape (M,)
            ground_truth: Ground truth voxel values of shape (M,)
            model: The GaussianVolumeModel (needed for regularization)
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # Main reconstruction loss
        losses['mse'] = self.reconstruction_loss(predicted, ground_truth)
        losses['total'] = losses['mse']
        
        if model is not None:
            # Sparsity regularization
            if self.use_sparsity:
                losses['sparsity'] = self.sparsity_loss(model.weights)
                losses['total'] = losses['total'] + losses['sparsity']
            
            # Overlap regularization
            if self.use_overlap:
                covariance = model.gaussians.get_covariance_matrices()
                losses['overlap'] = self.overlap_loss(model.positions, covariance)
                losses['total'] = losses['total'] + losses['overlap']
            
            # Smoothness regularization
            if self.use_smoothness:
                losses['smoothness'] = self.smoothness_loss(
                    model.positions, model.weights, model.log_scales
                )
                losses['total'] = losses['total'] + losses['smoothness']
        
        return losses


def compute_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(predicted, ground_truth).item()
    if mse == 0:
        return float('inf')
    
    max_val = ground_truth.max().item()
    psnr = 20 * torch.log10(torch.tensor(max_val / (mse ** 0.5)))
    return psnr.item()
