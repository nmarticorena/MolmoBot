import torch
import numpy as np
from typing import Optional
from molmobot_spoc.architecture.action_spaces.binned_continuous import (
    BinnedContinuousActionSpace,
)
from molmobot_spoc.utils.logger_utils import setup_logger

logger = setup_logger("QuantileBasedBinnedActionSpace")


class QuantileBasedBinnedContinuousActionSpace(BinnedContinuousActionSpace):
    """
    Action space that discretizes continuous actions using quantile-based binning.

    Instead of uniform bins in [-1, 1], this creates bins based on the actual
    distribution of actions in the dataset. This ensures that bins are placed
    where the data actually exists, potentially improving representation.

    Args:
        num_bins: Number of bins per dimension (default: 256)
        action_dim: Dimension of the action vector
        chunk_size: Number of timesteps per action chunk (default: 8)
        bin_edges_per_dim: Optional pre-computed bin edges tensor of shape (action_dim, num_bins + 1).
                          If provided, the action space will use these directly instead of requiring fit_to_dataset().
        normalization_mins: Optional normalization minimum values (shape: action_dim)
        normalization_maxs: Optional normalization maximum values (shape: action_dim)
    """

    def __init__(
        self,
        num_bins: int,
        action_dim: int,
        chunk_size: int,
        bin_edges_per_dim: torch.Tensor,
        normalization_mins: torch.Tensor,
        normalization_maxs: torch.Tensor,
        **kwargs,
    ):
        super().__init__(
            num_bins, action_dim, chunk_size, normalization_mins, normalization_maxs
        )

        self.bin_edges_per_dim = bin_edges_per_dim
        self.bin_centers_per_dim = self.bin_edges_per_dim[:, 1:]  # take the right edge

    def _get_vocab_mask(self) -> torch.Tensor:
        """
        Create a mask for valid (non-ghost) bins plus padding token.

        Ghost bins occur when multiple bin edges are identical due to
        data concentration. For consecutive bins with the same edge value,
        only the FIRST bin is kept valid, and the rest are masked.
        This ensures that torch.bucketize always assigns values to valid bins.

        Returns:
            Boolean tensor of shape [action_dim * chunk_size, num_bins + 1] where True indicates
            a valid bin and False indicates a ghost bin. The mask is repeated for each
            position in the chunk. The last position (padding token) is always True.
        """
        # Create base mask: (action_dim, num_bins)
        mask = torch.zeros(self.action_dim, self.num_bins, dtype=torch.bool)

        for dim in range(self.action_dim):
            edges = self.bin_edges_per_dim[dim]

            # For each bin, check if it's the first occurrence of its edge value
            # This handles consecutive duplicate edges by keeping only the first
            for i in range(self.num_bins):
                bin_start = edges[i]
                bin_end = edges[i + 1]

                # A bin is valid if:
                # 1. It has non-zero width (bin_end > bin_start), OR
                # 2. It's a zero-width bin but is the FIRST in a sequence of duplicates
                if bin_end > bin_start:
                    # Non-zero width bin - always valid
                    mask[dim, i] = True
                else:
                    # Zero-width bin - only valid if it's the first in the sequence
                    # Check if the previous bin has a different edge value
                    if i == 0 or edges[i] != edges[i - 1]:
                        mask[dim, i] = True
                    # else: subsequent duplicate, remains False (masked)

        # Add padding token column (always valid): (action_dim, num_bins) -> (action_dim, num_bins + 1)
        padding_col = torch.zeros(self.action_dim, 1, dtype=torch.bool)
        mask = torch.cat([mask, padding_col], dim=1)

        # Repeat for each position in chunk: (action_dim, num_bins + 1) -> (action_dim * chunk_size, num_bins + 1)
        # Pattern: [dim0, dim1, ..., dimN, dim0, dim1, ..., dimN, ...]
        mask_repeated = mask.repeat(self.chunk_size, 1)

        return mask_repeated

    def _value_to_bin(self, values: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values in [-1, 1] to bin indices using quantile-based bins.

        Args:
            values: Tensor of continuous values in [-1, 1]
                    Shape: (..., action_dim)

        Returns:
            Tensor of bin indices (0 to num_bins-1)
        """
        # Move bin edges to same device as values
        bin_edges = self.bin_edges_per_dim.to(values.device)

        # Get shape info
        original_shape = values.shape
        values_flat = values.reshape(-1, self.action_dim)

        # Bin each dimension independently
        bins = torch.zeros_like(values_flat, dtype=torch.long)
        for dim in range(self.action_dim):
            dim_values = values_flat[:, dim]
            dim_edges = bin_edges[dim, 1:-1]  # Exclude first and last for bucketize
            bins[:, dim] = torch.bucketize(dim_values, dim_edges)

        # Reshape back to original shape
        bins = bins.reshape(original_shape)
        return bins

    def _bin_to_value(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Convert bin indices back to continuous values using quantile-based bin centers.

        Args:
            bins: Tensor of bin indices (0 to num_bins-1)
                  Shape: (..., action_dim)

        Returns:
            Tensor of continuous values (bin centers) in [-1, 1]
        """
        # Move bin centers to same device as bins
        bin_centers = self.bin_centers_per_dim.to(bins.device)

        # Get shape info
        original_shape = bins.shape
        bins_flat = bins.reshape(-1, self.action_dim)

        # Handle padding token by clamping
        valid_bins = torch.clamp(bins_flat, 0, self.num_bins - 1)

        # Look up bin centers for each dimension
        values = torch.zeros_like(valid_bins, dtype=torch.float32)
        for dim in range(self.action_dim):
            dim_bins = valid_bins[:, dim]
            values[:, dim] = bin_centers[dim, dim_bins]

        # Reshape back to original shape
        values = values.reshape(original_shape)
        return values
