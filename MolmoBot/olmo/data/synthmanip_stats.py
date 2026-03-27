"""Normalization-stats computation for SynthManip actions and states.

The four ``_process_*`` functions are module-level so they can be pickled
by multiprocessing.Pool.  The three ``compute_*`` helpers wrap them with
the aggregation logic and are called by SynthmanipDataset.
"""

import json
import logging
import random
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from olmo.data.synthmanip_utils import _decode_h5_string, _open_h5_with_retry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiprocessing worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _process_trajectory_stats(task: Tuple) -> Optional[Tuple[Dict, Dict]]:
    """Worker: per-trajectory min/max statistics."""
    file_path, traj_idx, action_keys, action_move_group_names = task
    try:
        with _open_h5_with_retry(file_path) as f:
            traj_key = f"traj_{traj_idx}"
            stats_data = f["stats"]
            mins_dict: Dict[str, Any] = {}
            maxs_dict: Dict[str, Any] = {}
            for move_group in action_move_group_names:
                action_key = "joint_pos" if move_group == "gripper" else action_keys[move_group]
                stats_str = _decode_h5_string(stats_data[traj_key]['actions'][action_key])
                stats = json.loads(stats_str)
                if move_group in stats:
                    mins_dict[move_group] = stats[move_group]["min"]
                    maxs_dict[move_group] = stats[move_group]["max"]
            return mins_dict, maxs_dict
    except (OSError, KeyError):
        return None


def _process_trajectory_mean_std_stats(task: Tuple) -> Optional[Tuple[Dict, Dict]]:
    """Worker: per-trajectory mean/std statistics."""
    file_path, traj_idx, action_keys, action_move_group_names = task
    try:
        with _open_h5_with_retry(file_path) as f:
            traj_key = f"traj_{traj_idx}"
            stats_data = f["stats"]
            means_dict: Dict[str, Any] = {}
            stds_dict: Dict[str, Any] = {}
            for move_group in action_move_group_names:
                action_key = action_keys[move_group]
                stats_str = _decode_h5_string(stats_data[traj_key][action_key])
                stats = json.loads(stats_str)
                if move_group in stats:
                    means_dict[move_group] = stats[move_group]["mean"]
                    stds_dict[move_group] = stats[move_group]["std"]
            return means_dict, stds_dict
    except (OSError, KeyError):
        return None


def _process_sample_actions(task: Tuple) -> Optional[np.ndarray]:
    """Worker: collect all action timesteps from a trajectory for quantile computation.

    Excludes padding (frame 0) and done action (last frame).
    """
    file_path, traj_idx, action_move_group_names, action_spec, action_keys = task

    try:
        with _open_h5_with_retry(file_path) as f:
            traj_key = f"traj_{traj_idx}"
            action_data = f[traj_key]["actions"]

            unique_action_keys = list(set(action_keys.values()))
            # Also decode joint_pos when joint_pos_rel is needed (gripper fallback)
            if "joint_pos_rel" in unique_action_keys and "joint_pos" not in unique_action_keys:
                if "joint_pos" in action_data:
                    unique_action_keys.append("joint_pos")

            all_decoded: Dict[str, Any] = {}
            for key in unique_action_keys:
                if key not in action_data:
                    continue
                dataset = action_data[key]
                is_json_bytes = (dataset.dtype == np.uint8 and len(dataset.shape) == 2)
                if not is_json_bytes and dataset.dtype.kind in ('f', 'i', 'u'):
                    decoded = dataset[:].tolist()
                else:
                    decoded = []
                    for i in range(dataset.shape[0]):
                        byte_array = dataset[i]
                        json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                        decoded.append(json.loads(json_string))
                # Exclude padding (index 0) and done action (last)
                all_decoded[key] = decoded[1:-1] if len(decoded) > 2 else []

            if not all_decoded:
                return None
            first_key = unique_action_keys[0]
            if first_key not in all_decoded or not all_decoded[first_key]:
                return None

            num_steps = len(all_decoded[first_key])
            all_actions = []

            for i in range(num_steps):
                action_vec = []
                for move_group in action_move_group_names:
                    action_key = action_keys[move_group]
                    if action_key not in all_decoded:
                        action_vec.append(np.zeros(action_spec[move_group], dtype=np.float32))
                        continue

                    frame_data = all_decoded[action_key][i]

                    if isinstance(frame_data, dict):
                        if move_group in frame_data:
                            val = np.array(frame_data[move_group], dtype=np.float32)
                            action_vec.append(val[:action_spec[move_group]])
                        elif action_key == "joint_pos_rel" and "joint_pos" in all_decoded:
                            joint_pos_frame = all_decoded["joint_pos"][i]
                            if isinstance(joint_pos_frame, dict) and move_group in joint_pos_frame:
                                val = np.array(joint_pos_frame[move_group], dtype=np.float32)
                                action_vec.append(val[:action_spec[move_group]])
                            else:
                                action_vec.append(np.zeros(action_spec[move_group], dtype=np.float32))
                        else:
                            action_vec.append(np.zeros(action_spec[move_group], dtype=np.float32))
                    elif isinstance(frame_data, (list, tuple)):
                        action_vec.append(np.array(frame_data, dtype=np.float32))
                    else:
                        action_vec.append(np.array(frame_data, dtype=np.float32))

                all_actions.append(np.concatenate(action_vec))

            return np.array(all_actions, dtype=np.float32) if all_actions else None

    except Exception:
        return None


def _process_trajectory_state_stats(task: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Worker: per-trajectory state min/max statistics."""
    file_path, traj_idx, action_move_group_names, state_spec, state_indices, gripper_representation_count = task
    try:
        with _open_h5_with_retry(file_path) as f:
            traj_key = f"traj_{traj_idx}"
            try:
                qpos_data = f[traj_key]["obs"]["agent"]["qpos"]
                if not (qpos_data.dtype == np.uint8 and len(qpos_data.shape) == 2):
                    return None

                all_states = []
                for frame_idx in range(qpos_data.shape[0]):
                    byte_array = qpos_data[frame_idx]
                    json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                    qpos_dict = json.loads(json_string)
                    state_vec = []
                    for move_group in action_move_group_names:
                        if move_group in qpos_dict:
                            val = qpos_dict[move_group]
                            if "gripper" in move_group:
                                val = val[:gripper_representation_count]
                            elif move_group in state_indices:
                                val = [val[i] for i in state_indices[move_group]]
                            state_vec.extend(val if isinstance(val, (list, tuple)) else [val])
                        else:
                            state_vec.extend([0.0] * state_spec[move_group])
                    all_states.append(state_vec)

                if all_states:
                    arr = np.array(all_states, dtype=np.float32)
                    return np.min(arr, axis=0), np.max(arr, axis=0)
            except KeyError:
                pass
        return None
    except (OSError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Aggregation helpers (called by SynthmanipDataset)
# ---------------------------------------------------------------------------

def compute_action_normalization_stats(
    dataset: Any,
    num_workers: Optional[int] = None,
    mode: str = "min_max",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute action normalization stats for a SynthmanipDataset.

    Args:
        dataset: SynthmanipDataset instance (duck-typed to avoid circular import).
        num_workers: Parallel workers (defaults to cpu_count).
        mode: "min_max", "mean_std", or "quantile".
        **kwargs: For quantile mode: lower_quantile, upper_quantile, max_samples.

    Returns:
        (low, high) arrays of shape (action_dim,).
    """
    if mode == "quantile":
        lower_q = kwargs.get("lower_quantile", 0.01)
        upper_q = kwargs.get("upper_quantile", 0.99)
        max_samples = kwargs.get("max_samples", 10000)
        l, u = _compute_quantile_stats(dataset, lower_q, upper_q, max_samples, num_workers)
        return l.numpy(), u.numpy()

    if num_workers is None:
        num_workers = cpu_count()

    log.info(f"Computing action normalization stats ({mode}) with {num_workers} workers...")

    tasks = [
        (str(dataset._files[fi]), ti, dataset.action_keys, dataset.action_move_group_names)
        for fi, ti in (dataset._get_file_and_traj_idx(gi) for gi in dataset.traj_indices)
    ]

    worker_fn = _process_trajectory_stats if mode == "min_max" else _process_trajectory_mean_std_stats

    with Pool(num_workers) as pool:
        results = pool.map(worker_fn, tasks)

    all_first = {mg: [] for mg in dataset.action_move_group_names}
    all_second = {mg: [] for mg in dataset.action_move_group_names}

    for result in results:
        if result is None:
            continue
        first_dict, second_dict = result
        for mg in dataset.action_move_group_names:
            if mg in first_dict:
                all_first[mg].append(first_dict[mg])
                all_second[mg].append(second_dict[mg])

    agg_first, agg_second = [], []
    for mg in dataset.action_move_group_names:
        if all_first[mg]:
            arr_f = np.array(all_first[mg])
            arr_s = np.array(all_second[mg])
            if mode == "min_max":
                agg_first.append(np.min(arr_f, axis=0))
                agg_second.append(np.max(arr_s, axis=0))
            else:
                agg_first.append(np.mean(arr_f, axis=0))
                agg_second.append(np.mean(arr_s, axis=0))
        else:
            log.warning(f"No data for move group '{mg}', using zeros")
            agg_first.append(np.zeros(dataset.action_spec[mg]))
            agg_second.append(np.zeros(dataset.action_spec[mg]))

    log.info(f"Computed normalization stats from {sum(1 for r in results if r)} trajectories")
    return np.concatenate(agg_first), np.concatenate(agg_second)


def _compute_quantile_stats(
    dataset: Any,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    max_samples: int = 10000,
    num_workers: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quantile normalization stats over a random sample of trajectories."""
    if num_workers is None:
        num_workers = cpu_count()
    if num_workers == 0:
        num_workers = 16

    log.info(f"Computing quantile stats (q={lower_quantile},{upper_quantile}) with {num_workers} workers...")

    num_samples = min(max_samples, len(dataset.traj_indices))
    sampled_positions = random.sample(range(len(dataset.traj_indices)), num_samples)
    log.info(f"Sampling actions from {num_samples} trajectories")

    tasks = []
    for pos in sampled_positions:
        gi = dataset.traj_indices[pos]
        fi, ti = dataset._get_file_and_traj_idx(gi)
        tasks.append((str(dataset._files[fi]), ti, dataset.action_move_group_names, dataset.action_spec, dataset.action_keys))

    with Pool(num_workers) as pool:
        results = pool.map(_process_sample_actions, tasks)

    all_actions = [r for r in results if r is not None]
    if not all_actions:
        raise ValueError("No actions could be collected for quantile computation")

    all_actions_np = np.concatenate(all_actions, axis=0)
    log.info(f"Collected {all_actions_np.shape[0]} action timesteps (dim={all_actions_np.shape[1]})")

    all_actions_t = torch.from_numpy(all_actions_np).float()
    return (
        torch.quantile(all_actions_t, lower_quantile, dim=0),
        torch.quantile(all_actions_t, upper_quantile, dim=0),
    )


def compute_state_normalization_stats(
    dataset: Any,
    num_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute state min/max normalization stats for a SynthmanipDataset.

    Args:
        dataset: SynthmanipDataset instance.
        num_workers: Parallel workers (defaults to cpu_count).

    Returns:
        (global_min, global_max) arrays of shape (state_dim,).
    """
    if num_workers is None:
        num_workers = cpu_count()
    if num_workers == 0:
        num_workers = 16

    tasks = [
        (
            str(dataset._files[fi]),
            ti,
            dataset.action_move_group_names,
            dataset.state_spec,
            dataset.state_indices,
            dataset.config.gripper_representation_count,
        )
        for fi, ti in (dataset._get_file_and_traj_idx(gi) for gi in dataset.traj_indices)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_process_trajectory_state_stats, tasks)

    all_mins = [r[0] for r in results if r is not None]
    all_maxs = [r[1] for r in results if r is not None]

    if not all_mins:
        log.warning("No state data found, using zeros")
        return np.zeros(dataset.state_dim), np.zeros(dataset.state_dim)

    log.info(f"Computed state normalization stats from {len(all_mins)} trajectories")
    return np.minimum.reduce(all_mins), np.maximum.reduce(all_maxs)
