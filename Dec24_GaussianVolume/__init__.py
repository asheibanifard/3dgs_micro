"""
Dec24 Gaussian Volume - Package initialization

Gaussian-Based Volume Data Representation
Implementation based on [21 Dec. 24] Algorithm from research proposal.

Main components:
- GaussianBasisFunctions: Core Gaussian basis functions
- GaussianVolumeModel: Complete model for volume representation
- TotalLoss: Combined loss function with regularization
- GaussianVolumeTrainer: Training loop implementation
"""

from .gaussian_model import (
    GaussianBasisFunctions,
    GaussianVolumeModel
)

from .losses import (
    ReconstructionLoss,
    SparsityRegularization,
    OverlapRegularization,
    SmoothnessRegularization,
    TotalLoss,
    compute_psnr
)

from .trainer import (
    VolumeDataSampler,
    GaussianVolumeTrainer,
    create_trainer
)

__all__ = [
    # Models
    'GaussianBasisFunctions',
    'GaussianVolumeModel',
    # Losses
    'ReconstructionLoss',
    'SparsityRegularization',
    'OverlapRegularization',
    'SmoothnessRegularization',
    'TotalLoss',
    'compute_psnr',
    # Training
    'VolumeDataSampler',
    'GaussianVolumeTrainer',
    'create_trainer',
]
