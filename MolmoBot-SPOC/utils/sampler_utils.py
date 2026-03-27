from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
import math
from collections import defaultdict
from molmobot_spoc.utils.logger_utils import setup_logger

logger = setup_logger("TaskTypeWeightedSampler")


class TaskTypeWeightedSampler(Sampler):
    """
    Custom sampler that handles large datasets (>2^24 samples) by grouping samples by task type.

    Uses two-stage sampling (pick task type, then pick sample within it) to avoid
    storing per-sample weights for large datasets and the torch.multinomial 2^24 limit.

    """

    def __init__(
        self,
        indices,
        task_types,
        task_sampling_weights,
        num_samples=None,
        seed=0,
    ):
        """
        Args:
            indices: List of sample indices in the dataset
            task_types: List of task type strings corresponding to each index
            task_sampling_weights: Dict mapping task_type to weight. Weights are used directly as
                relative probabilities (normalized to sum to 1). For example, weights {A: 35.0, B: 20.0}
            num_samples: Number of samples to draw per epoch (total across all ranks). If None, defaults to len(indices)
            seed: Random seed for reproducibility
        """
        if len(indices) != len(task_types):
            raise ValueError(
                f"indices and task_types must have the same length, got {len(indices)} and {len(task_types)}"
            )

        self.indices = np.asarray(indices)
        self.task_types = task_types
        self.seed = seed
        self.epoch = 0

        # num_samples is total samples across all ranks per epoch
        if num_samples is None:
            num_samples = len(indices)
        self.num_samples = num_samples
        self.total_size = num_samples

        # Group positions (0 to len(indices)-1) by task type
        # We work with positions internally, then map back to actual indices
        self.task_type_to_positions = defaultdict(list)
        for pos, (idx, task_type) in enumerate(zip(indices, task_types)):
            self.task_type_to_positions[task_type].append(pos)

        # Store the actual indices array for mapping positions back to indices
        self.indices_array = np.asarray(indices)

        # Convert to numpy arrays for efficient sampling
        task_type_list = []
        task_type_sizes = []

        for task_type, positions_list in sorted(self.task_type_to_positions.items()):
            positions_array = np.asarray(positions_list)
            self.task_type_to_positions[task_type] = positions_array
            task_type_list.append(task_type)
            task_type_sizes.append(len(positions_array))

        self.task_type_list = task_type_list
        self.task_type_sizes = np.array(task_type_sizes)

        # Cache position arrays for faster iteration (avoid recreating list every time)
        self.position_arrays = [
            self.task_type_to_positions[task_type] for task_type in task_type_list
        ]

        # Build task type weights (normalized probabilities)
        # Weights are used directly as relative probabilities, NOT multiplied by dataset size
        task_type_weights = np.array(
            [task_sampling_weights.get(task_type, 1.0) for task_type in task_type_list],
            dtype=np.float64,
        )

        # Normalize to probabilities
        total_weight = task_type_weights.sum()
        if total_weight > 0:
            self.task_type_probs = task_type_weights / total_weight
        else:
            # Fallback to uniform
            self.task_type_probs = np.ones(len(task_type_list)) / len(task_type_list)

        task_type_counts = {
            task_type: size for task_type, size in zip(task_type_list, task_type_sizes)
        }
        logger.info(
            f"TaskTypeWeightedSampler initialized with {len(task_type_list)} task types"
        )
        logger.info(
            f"Task type probabilities: {dict(zip(task_type_list, self.task_type_probs))}"
        )
        logger.info(f"Task type sample counts: {task_type_counts}")
        logger.info(f"Total samples per epoch: {self.num_samples}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Stage 1: pick which task type each sample comes from
        task_type_indices = rng.choice(
            len(self.task_type_probs),
            size=self.total_size,
            replace=True,
            p=self.task_type_probs,
        )

        # Stage 2: uniformly pick a position within each chosen task type
        positions = np.empty(self.total_size, dtype=np.int64)

        # Vectorized sampling: for each task type, sample all needed positions at once
        # Use cached position_arrays instead of recreating the list
        for task_type_idx in range(len(self.task_type_list)):
            mask = task_type_indices == task_type_idx
            count = int(mask.sum())
            if count > 0:
                # Sample uniformly from positions in this task type
                positions_array = self.position_arrays[task_type_idx]
                # Use choice with replace=True for uniform sampling (slightly faster than integers)
                sampled_positions = rng.choice(
                    len(positions_array), size=count, replace=True
                )
                positions[mask] = positions_array[sampled_positions]

        # Map positions back to actual dataset indices
        indices = self.indices_array[positions]

        return iter(indices.tolist())

    def __len__(self):
        # Return total_size (total across all ranks)
        # Lightning's DistributedSampler will divide this by num_replicas
        return self.total_size

    def set_epoch(self, epoch: int):
        """Call this at the start of each epoch to get a different shuffle."""
        self.epoch = epoch
