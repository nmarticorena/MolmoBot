"""
Binned Continuous Action Space

A simple tokenizer that discretizes continuous actions using uniform binning.
Each action dimension is independently mapped to one of N bins.

Token sequence: [bin_for_dim0_t0, bin_for_dim1_t0, ..., bin_for_dimD_t0,
                 bin_for_dim0_t1, ..., bin_for_dimD_tT]

Total tokens per chunk = chunk_size * action_dim
"""

import torch
from typing import Optional, Union, List

from molmobot_spoc.architecture.action_spaces.abstract import AbstractActionSpace
from molmobot_spoc.utils.logger_utils import setup_logger

logger = setup_logger("BinnedActionSpace")


class BinnedContinuousActionSpace(AbstractActionSpace):
    """
    Action space that discretizes continuous actions using uniform binning.

    Each dimension of the action is independently binned into `num_bins` discrete values.
    The bin centers are uniformly distributed in [-1, 1] after normalization.

    Args:
        num_bins: Number of bins per dimension (default: 256)
        action_dim: Dimension of the action vector
        chunk_size: Number of timesteps per action chunk (default: 8)
    """

    def __init__(
        self,
        num_bins: int,
        action_dim: int,
        chunk_size: int,
        normalization_mins: torch.Tensor,
        normalization_maxs: torch.Tensor,
        **kwargs,
    ):
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.is_tokenized = True
        self.is_continuous_actions = True
        self.padding_token = self.num_bins

        # Token sequence length = chunk_size * action_dim
        self.max_token_seq_len = chunk_size * action_dim

        # Normalization stats (to be set from dataset)
        self.normalization_mins = normalization_mins
        self.normalization_maxs = normalization_maxs

        # Compute bin edges and centers for [-1, 1] range
        # Bins: [-1, -1+2/N), [-1+2/N, -1+4/N), ..., [1-2/N, 1]
        self.bin_edges = torch.linspace(-1, 1, num_bins + 1)
        self.bin_centers = self.bin_edges[1:]  # take the right edge

        logger.info(
            f"Initialized BinnedContinuousActionSpace with {num_bins} bins, "
            f"action_dim={action_dim}, chunk_size={chunk_size}, "
            f"token_seq_len={self.max_token_seq_len}"
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions to [-1, 1] range using min-max normalization.

        Args:
            actions: Tensor of shape (batch_size, chunk_size, action_dim)

        Returns:
            Normalized actions in [-1, 1] range
        """
        if self.normalization_maxs is not None and self.normalization_mins is not None:
            mins = self.normalization_mins.view(1, 1, -1).to(actions.device)
            maxs = self.normalization_maxs.view(1, 1, -1).to(actions.device)
        else:
            raise ValueError(
                "Please set the normalization mins and maxs in the action space."
            )

        # Normalize to [0, 1] then scale to [-1, 1]
        normalized = 2 * (actions - mins) / (maxs - mins + 1e-8) - 1
        # Clamp to ensure we stay in [-1, 1]
        return torch.clamp(normalized, -1, 1)

    def denormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions from [-1, 1] back to original range.

        Args:
            actions_norm: Normalized actions in [-1, 1] range
                          Shape: (batch_size, window_size, chunk_size, action_dim)

        Returns:
            Actions in original range
        """
        if self.normalization_mins is None or self.normalization_maxs is None:
            logger.warning("No normalization stats available, returning as-is")
            return actions_norm

        # Move normalization stats to the same device as input
        mins = self.normalization_mins.to(actions_norm.device)
        maxs = self.normalization_maxs.to(actions_norm.device)
        actions = ((actions_norm + 1) / 2) * (maxs - mins) + mins
        return actions

    def _value_to_bin(self, values: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values in [-1, 1] to bin indices.

        Args:
            values: Tensor of continuous values in [-1, 1]

        Returns:
            Tensor of bin indices (0 to num_bins-1)
        """
        # Map [-1, 1] to [0, num_bins-1]
        # Use bucketize for efficient binning
        bin_edges = self.bin_edges.to(values.device)
        # bucketize returns index where value would be inserted
        # Subtract 1 and clamp to get bin index
        bins = torch.bucketize(values, bin_edges[1:-1])  # This gives 0 to num_bins-1
        return bins.long()

    def _bin_to_value(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Convert bin indices back to continuous values (bin centers).

        Args:
            bins: Tensor of bin indices (0 to num_bins-1)

        Returns:
            Tensor of continuous values (bin centers) in [-1, 1]
        """
        bin_centers = self.bin_centers.to(bins.device)
        # Handle padding token by clamping
        valid_bins = torch.clamp(bins, 0, self.num_bins - 1)
        return bin_centers[valid_bins]

    def tokenize_actions(
        self,
        actions: torch.Tensor,
        actions_is_pad: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize continuous actions using binning.

        Args:
            actions: Continuous actions of shape (batch_size, chunk_size, action_dim)
            actions_is_pad: Padding mask of shape (batch_size, chunk_size)
            return_attention_mask: Whether to return attention mask

        Returns:
            tokens: Discrete token IDs of shape (batch_size, chunk_size * action_dim)
            attention_mask: (optional) Attention mask of shape (batch_size, chunk_size * action_dim)
        """
        batch_size = actions.shape[0]

        # Normalize to [-1, 1]
        actions_norm = self.normalize_actions(actions)

        # Convert to bins: (batch_size, chunk_size, action_dim) -> (batch_size, chunk_size * action_dim)
        tokens = self._value_to_bin(actions_norm)
        tokens = tokens.reshape(batch_size, -1)  # Flatten chunk_size * action_dim

        # Set padding tokens where actions_is_pad is True
        # Expand actions_is_pad from (batch_size, chunk_size) to (batch_size, chunk_size, action_dim)
        # then flatten to (batch_size, chunk_size * action_dim)
        actions_is_pad_expanded = actions_is_pad.unsqueeze(-1).expand(
            -1, -1, self.action_dim
        )
        actions_is_pad_flat = actions_is_pad_expanded.reshape(batch_size, -1)
        tokens[actions_is_pad_flat] = self.padding_token

        return tokens

    def decode_actions(
        self,
        tokens: torch.Tensor,
        action_dim: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode tokens back to continuous actions.

        Args:
            tokens: Discrete token IDs of shape (batch_size, seq_len)
                    where seq_len = chunk_size * action_dim
            action_dim: Expected action dimension (will use self.action_dim if not provided)
            chunk_size: Expected chunk size (will use self.chunk_size if not provided)

        Returns:
            Continuous actions of shape (batch_size, chunk_size, action_dim)
        """
        action_dim = action_dim or self.action_dim
        chunk_size = chunk_size or self.chunk_size

        batch_size, seq_len = tokens.shape

        # Track padding positions before _bin_to_value clamps them away
        is_padding = tokens == self.padding_token  # (batch_size, seq_len)

        # Convert bins to values: (batch_size, seq_len) -> (batch_size, seq_len)
        values_norm = self._bin_to_value(tokens)

        # Reshape to (batch_size, chunk_size, action_dim)
        values_norm = values_norm.reshape(batch_size, chunk_size, action_dim)

        # Denormalize
        actions = self.denormalize_actions(values_norm)

        # Zero out padding positions so the robot stays still
        is_padding = is_padding.reshape(batch_size, chunk_size, action_dim)
        actions = actions.masked_fill(is_padding, 0.0)

        return actions

    def _get_singleton_action(
        self,
        action_string: str = "",
        action_quantity: float = None,
    ):
        raise NotImplementedError(
            "Singleton actions not supported for binned continuous action space."
        )

    def get_action(
        self,
        action_strings: Union[str, List[str]] = None,
        action_quantities: Union[float, List[float]] = None,
    ):
        raise NotImplementedError(
            "get_action not supported for binned continuous action space."
        )

    def get_num_actions(self) -> int:
        """
        Returns the vocabulary size (num_bins + 1 for padding).
        """
        return self.num_bins + 1

    def get_action_list(self) -> List[str]:
        """
        Return list of bin names.
        """
        return [f"bin_{i}" for i in range(self.get_num_actions())]
