import os
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np

from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.synthmanip_dataset import synthmanip_config_registry, SynthmanipDatasetConfig


def compute_state_action_normalization_stats(
        data_paths: List[str],
        camera_names: List[str],
        action_move_group_names: List[str],
        action_spec: dict,
        action_keys: dict,
        action_horizon: int,
        input_window_size: int = 1,
        num_workers: int = 8,
        state_spec: Optional[dict] = None,
        state_indices: Optional[dict] = None,
) -> tuple[dict, str, str]:
    from olmo.data.synthmanip_dataset import SynthmanipDataset

    """Compute normalization statistics for actions (quantile) and state (min_max)."""

    total_max_samples = 10000
    samples_per_dataset = max(1, total_max_samples // len(data_paths))
    print(f"Computing norm stats: {total_max_samples} total samples, {samples_per_dataset} per dataset ({len(data_paths)} datasets)", flush=True)

    state_minimum_list = []
    state_maximum_list = []

    action_minimum_list = []
    action_maximum_list = []

    for data_path in data_paths:
        print(f"Computing state stats from: {data_path}")
        config = SynthmanipDatasetConfig(
            data_path=data_path,
            camera_names=camera_names,
            action_move_group_names=action_move_group_names,
            action_spec=action_spec,
            action_keys=action_keys,
            state_spec=state_spec,
            state_indices=state_indices,
            action_horizon=action_horizon,
            input_window_size=input_window_size,
            split="train",
        )
        dataset = SynthmanipDataset(config)

        action_min, action_max = dataset.get_action_normalization_stats(num_workers=num_workers, mode="quantile", max_samples=samples_per_dataset)
        print(f"  action stats shape: min={action_min.shape}, max={action_max.shape} from {data_path}", flush=True)
        action_minimum_list.append(action_min)
        action_maximum_list.append(action_max)

        state_min, state_max = dataset.get_state_normalization_stats(num_workers=num_workers)
        print(f"  state stats shape: min={state_min.shape}, max={state_max.shape} from {data_path}", flush=True)
        state_minimum_list.append(state_min)
        state_maximum_list.append(state_max)

    # Aggregate: take global min and max across all datasets
    print(f"Aggregating stats across {len(action_minimum_list)} datasets...", flush=True)
    for i, (amin, smin) in enumerate(zip(action_minimum_list, state_minimum_list)):
        print(f"  dataset {i}: action_shape={amin.shape}, state_shape={smin.shape}", flush=True)

    global_state_min = np.minimum.reduce(state_minimum_list)
    global_state_max = np.maximum.reduce(state_maximum_list)

    global_action_min = np.minimum.reduce(action_minimum_list)
    global_action_max = np.maximum.reduce(action_maximum_list)

    stats = {"observation.state": {"min": global_state_min.tolist(), "max": global_state_max.tolist()},
            "action": {"q01": global_action_min.tolist(), "q99": global_action_max.tolist()}}

    return stats, "quantiles", "min_max"


def compute_normalization_stats(
    data_paths: List[str],
    camera_names: List[str],
    action_move_group_names: List[str],
    action_spec: dict,
    action_keys: dict,
    action_horizon: int,
    input_window_size: int = 1,
    num_workers: int = 8,
    save_path: Optional[str] = None,
) -> RobotProcessorConfig:
    """Compute quantile-based normalization statistics from SynthManip-format datasets.

    Uses 1st and 99th percentiles (q01/q99) for robust normalization that is less
    sensitive to outliers than min/max.

    Args:
        data_paths: List of paths to task type directories. Each must contain train/.
    """
    from olmo.data.synthmanip_dataset import SynthmanipDataset

    all_q01 = []
    all_q99 = []

    for data_path in data_paths:
        print(f"Computing quantile stats from: {data_path}")
        config = SynthmanipDatasetConfig(
            data_path=data_path,
            camera_names=camera_names,
            action_move_group_names=action_move_group_names,
            action_spec=action_spec,
            action_keys=action_keys,
            action_horizon=action_horizon,
            input_window_size=input_window_size,
            split="train",
        )
        dataset = SynthmanipDataset(config)

        q01, q99 = dataset.get_action_normalization_stats(num_workers=num_workers, mode="quantile")
        all_q01.append(q01)
        all_q99.append(q99)

    # Aggregate: take global min of q01 and max of q99 across all datasets
    global_q01 = np.minimum.reduce(all_q01)
    global_q99 = np.maximum.reduce(all_q99)

    # Build RobotProcessorConfig with quantile stats
    stats = {
        "action": {
            "q01": global_q01.tolist(),
            "q99": global_q99.tolist(),
        }
    }

    proc_config = RobotProcessorConfig.from_stats(
        stats_by_repo={"synthmanip": stats},
        default_repo_id="synthmanip",
        action_norm_mode="quantiles",
        state_norm_mode="quantiles",
    )

    if save_path:
        print(f"Saving normalization stats to {save_path}")
        proc_config.save(save_path)

    return proc_config


def get_synthmanip_training_data(
    data_paths: List[str],
    camera_names: List[str],
    action_move_group_names: List[str],
    action_spec: dict,
    action_keys: dict,
    action_horizon: int,
    input_window_size: int = 1,
    obs_step_delta: int = 1,
    robot_processor_config: Optional[RobotProcessorConfig] = None,
    weighted_sampling: bool = False,
    randomize_prompts: bool = False,
    dataset_sample_rates: Optional[List[float]] = None,
    furthest_camera_prob: float = 0.0,
    use_point_prompts: bool = False,
    point_prompt_camera: str = "head_camera",
    conditioning_frame: Union[int, str] = "random_first_10",
    cameras_to_warp: Optional[List[str]] = None,
    max_points_in_conditioning_frame: int = 1,
    use_point_prompts_per_dataset: Optional[List[bool]] = None,
    require_val: bool = True,
    state_spec: Optional[dict] = None,
    state_indices: Optional[dict] = None,
    max_exo_views: int = 1,
) -> Optional[Dict[str, float]]:
    """Build training data mixture for SynthManip-format dataset.

    Registers the dataset configurations with the synthmanip_config_registry so that
    build_synthmanip_dataset can retrieve them when creating the datasets.

    Each data_path should point to a task type directory containing train/ and val/ subdirectories.
    Multiple data_paths are aggregated into a single training mixture.

    Args:
        data_paths: List of paths to task type directories. Each must contain train/ (and val/ unless require_val=False).
        require_val: Whether to require val/ subdirs in data_paths. False when --val_data_path is used.

    Returns:
        Dict of dataset names to sample rates for training mixture.
    """
    from olmo.data.data_loader import DatasetWithArgs

    # Validate all data paths first
    for data_path in data_paths:
        validate_data_path(data_path, require_val=require_val)

    weight_config = None
    if weighted_sampling:
        weight_config = {
            'lookback_window': 0,
            'lookahead_window': 10,
            'final_grasp_weight': 2.0,
            'failed_grasp_weight': 0.5,
            'release_after_failed_grasp_weight': 3.0,
            'gripper_threshold': 127.5,
            'go_home_weight': 2.0,
            'go_home_start_frames': 5,
            'go_home_end_frames': 20,
            'verbose': False,  # Set to True for detailed debug output
        }

    if dataset_sample_rates:
        assert len(dataset_sample_rates) == len(data_paths), "dataset_sample_rate must have same length as data_paths"
    if use_point_prompts_per_dataset:
        assert len(use_point_prompts_per_dataset) == len(data_paths), "use_point_prompts_per_dataset must have same length as data_paths"

    # Create dataset configurations for each data path
    # datasets = []
    action_mixture = {}
    for i, data_path in enumerate(data_paths):
        use_pts = use_point_prompts_per_dataset[i] if use_point_prompts_per_dataset else use_point_prompts
        config = SynthmanipDatasetConfig(
            data_path=os.path.abspath(data_path),
            camera_names=camera_names,
            action_move_group_names=action_move_group_names,
            action_spec=action_spec,
            action_keys=action_keys,
            state_spec=state_spec,
            state_indices=state_indices,
            action_horizon=action_horizon,
            input_window_size=input_window_size,
            obs_step_delta=obs_step_delta,
            robot_processor_config=robot_processor_config,
            weighted_sampling=weighted_sampling,
            weight_config=weight_config,
            randomize_prompts=randomize_prompts,
            furthest_camera_prob=furthest_camera_prob,
            use_point_prompts=use_pts,
            point_prompt_camera=point_prompt_camera,
            conditioning_frame=conditioning_frame,
            cameras_to_warp=cameras_to_warp or [],
            max_points_in_conditioning_frame=max_points_in_conditioning_frame,
            max_exo_views=max_exo_views,
        )

        # Register the configuration with a unique name
        dataset_name = f"synthmanip/task_{i}"
        synthmanip_config_registry.register(dataset_name, config)
        print(f"Registered dataset '{dataset_name}' for: {data_path} (point_prompts={use_pts})")

        #datasets.append(DatasetWithArgs(dataset_name=dataset_name))
        action_mixture[dataset_name] =  dataset_sample_rates[i] if dataset_sample_rates else 1.0

    return action_mixture


def validate_data_path(data_path: str, require_val: bool = True) -> None:
    """Validate that a data path contains required subdirectories.

    Args:
        data_path: Path to the task type directory (e.g., /path/to/SomeTaskConfig)
        require_val: Whether to require a val/ subdirectory. Set False when a separate
                     --val_data_path is provided.

    Raises:
        ValueError: If the path doesn't exist or is missing required subdirs
    """
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    if not path.is_dir():
        raise ValueError(f"Data path is not a directory: {data_path}")

    train_path = path / "train"

    missing = []
    if not train_path.exists():
        missing.append("train/")
    if require_val and not (path / "val").exists():
        missing.append("val/")

    if missing:
        raise ValueError(
            f"Data path '{data_path}' is missing required subdirectories: {', '.join(missing)}\n"
            f"Expected structure: {data_path}/train/house_*/*.h5"
            + (f" and {data_path}/val/house_*/*.h5" if require_val else "")
        )
