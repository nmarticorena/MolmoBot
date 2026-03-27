import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
import h5py
import numpy as np
from decord import VideoReader, cpu
import decord
import torch
from collections import defaultdict
import json
import random
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from molmobot_spoc.utils.constants.prompt_templates import DEFAULT_PROMPT_TEMPLATES
from molmobot_spoc.utils.logger_utils import setup_logger
from molmobot_spoc.utils.dataset_utils import pad_data

logger = setup_logger("Dataset")


def _find_samples_with_phases(task):
    """Find sample steps whose action chunks contain any of the target phases.

    For a sample at step s with action_chunk_size A,
    the action steps span [s + 1, s + A]. If any target phase appears at
    any of those action steps, the sample should be upsampled for that phase.

    Args:
        task: Tuple of (file_path, traj_idx, traj_length,
                        action_chunk_size, target_str_phases) where target_str_phases
                        is a list of phase name strings. The int encoding is resolved
                        per-file from policy_phases in obs_scene, since different
                        datasets can use different int mappings.

    Returns:
        Dict mapping each target phase (str) to a list of step indices whose action
        chunks contain that phase.
    """
    (
        file_path,
        traj_idx,
        traj_length,
        action_chunk_size,
        target_str_phases,
    ) = task
    try:
        with h5py.File(file_path, "r") as f:
            traj_key = f"traj_{traj_idx}"

            # Resolve str phase names -> ints using this file's policy_phases dict
            policy_phases = json.loads(f[traj_key]["obs_scene"].astype("T")[()])[
                "policy_phases"
            ]
            str_to_int = {
                p: policy_phases[p] for p in target_str_phases if p in policy_phases
            }
            int_to_str = {v: k for k, v in str_to_int.items()}
            target_phase_ints = list(str_to_int.values())

            phase_data = f[traj_key]["obs"]["extra"]["policy_phase"][:]

            if phase_data.dtype.kind in ("i", "f", "u"):
                phase_array = phase_data
            else:
                phase_array = np.zeros(len(phase_data), dtype=np.int32)
                for i in range(len(phase_data)):
                    byte_array = phase_data[i]
                    json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                    phase_array[i] = json.loads(json_string)

            result = {p: [] for p in target_str_phases}
            for step in range(traj_length):
                action_start = max(1, step + 1)
                action_end = min(step + action_chunk_size + 1, len(phase_array))
                if action_start >= len(phase_array):
                    continue
                chunk = phase_array[action_start:action_end]
                for phase_int in target_phase_ints:
                    if np.any(chunk == phase_int):
                        result[int_to_str[phase_int]].append(step)

            return result
    except Exception:
        return {p: [] for p in target_str_phases}


def _load_task_types_from_file(task):
    """Helper function for parallel processing of task types from a file.

    Args:
        task: Tuple of (file_path, traj_dict) where traj_dict maps traj_key to traj_length

    Returns:
        Dict mapping traj_idx to task_type, or None if error
    """
    file_path, traj_dict = task
    task_types_for_file = {}

    with h5py.File(file_path, "r") as f:
        for traj_key in traj_dict.keys():
            traj_idx = int(traj_key.split("_")[1])
            try:
                h5_traj_key = f"traj_{traj_idx}"
                obs_scene_data = f[h5_traj_key]["obs_scene"]
                obs_scene_str = obs_scene_data.astype("T")[()]
                obs_scene = json.loads(obs_scene_str)
                task_type = obs_scene.get("task_type", "unknown")
                task_types_for_file[traj_idx] = task_type
            except Exception:
                # Silently skip errors in worker processes
                task_types_for_file[traj_idx] = "unknown"

    return task_types_for_file


def _process_trajectory_stats(task):
    """Helper function for parallel processing of trajectory stats.

    Args:
        task: Tuple of (file_path, traj_idx, action_keys, action_move_group_names)

    Returns:
        Tuple of (mins_dict, maxs_dict) or None if error
    """
    file_path, traj_idx, action_keys, action_move_group_names = task

    try:
        with h5py.File(file_path, "r") as file:
            traj_key = f"traj_{traj_idx}"

            # Read stats from the file (all action types are under stats/traj_i/actions/)
            stats_data = file["stats"]

            mins_dict = {}
            maxs_dict = {}

            for move_group in action_move_group_names:
                # Get the appropriate action key for this move group
                action_key = action_keys[move_group]
                stats = json.loads(stats_data[traj_key][action_key].astype("T")[()])

                if move_group in stats:
                    mins_dict[move_group] = stats[move_group]["min"]
                    maxs_dict[move_group] = stats[move_group]["max"]

            return mins_dict, maxs_dict

    except (OSError, KeyError) as e:
        # Silently skip errors in worker processes
        return None


def _process_trajectory_mean_std_stats(task):
    """Helper function for parallel processing of trajectory mean/std stats.

    Args:
        task: Tuple of (file_path, traj_idx, action_keys, action_move_group_names)

    Returns:
        Tuple of (means_dict, stds_dict) or None if error
    """
    file_path, traj_idx, action_keys, action_move_group_names = task

    try:
        with h5py.File(file_path, "r") as file:
            traj_key = f"traj_{traj_idx}"

            # Read stats from the file (all action types are under stats/traj_i/actions/)
            stats_data = file["stats"]

            means_dict = {}
            stds_dict = {}

            for move_group in action_move_group_names:
                # Get the appropriate action key for this move group
                action_key = action_keys[move_group]
                stats = json.loads(stats_data[traj_key][action_key].astype("T")[()])

                if move_group in stats:
                    means_dict[move_group] = stats[move_group]["mean"]
                    stds_dict[move_group] = stats[move_group]["std"]

            return means_dict, stds_dict

    except (OSError, KeyError) as e:
        # Silently skip errors in worker processes
        return None


def _process_sample_actions(task):
    """Helper function for parallel processing of action samples for quantile computation.

    Excludes padding (first action) and done action (last action) from the returned data.

    Args:
        task: Tuple of (file_path, traj_idx, action_move_group_names, action_spec, action_keys)

    Returns:
        numpy array of actions or None if error
    """
    file_path, traj_idx, action_move_group_names, action_spec, action_keys = task

    # Debug: Check types
    if not isinstance(action_keys, dict):
        logger.debug(f"ERROR: action_keys is {type(action_keys)}, value: {action_keys}")
        logger.debug(f"Full task: {task}")
        return None

    try:
        with h5py.File(file_path, "r") as file:
            traj_key = f"traj_{traj_idx}"
            action_data = file[traj_key]["actions"]

            # Get unique action keys needed
            unique_action_keys = list(set(action_keys.values()))

            # Decode all required action keys and slice to exclude padding and done action
            all_decoded_data = {}
            for actions_key in unique_action_keys:
                key_data = action_data[actions_key]
                decoded_data = []
                for i in range(key_data.shape[0]):
                    byte_array = key_data[i]
                    json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                    trajectory_dict = json.loads(json_string)
                    decoded_data.append(trajectory_dict)

                # Slice to exclude first (padding) and last (done) actions: [1:-1]
                all_decoded_data[actions_key] = decoded_data[1:-1]

            # Collect all actions from the sliced data
            all_actions = []
            for i in range(len(all_decoded_data[unique_action_keys[0]])):
                action_vec = []
                for move_group in action_move_group_names:
                    # Get the appropriate action key for this move group
                    action_key = action_keys[move_group]
                    decoded_data = all_decoded_data[action_key]

                    try:
                        action_vec.append(
                            decoded_data[i][move_group][: action_spec[move_group]]
                        )
                    except (KeyError, IndexError):
                        action_vec.append(np.zeros(action_spec[move_group]))

                action_vec = np.concatenate(action_vec)
                all_actions.append(action_vec)

            if len(all_actions) == 0:
                return None

            # Return as numpy array: (num_actions, action_dim)
            return np.array(all_actions, dtype=np.float32)

    except Exception as e:
        import traceback

        logger.debug(f"Error processing {file_path}, traj {traj_idx}: {e}")
        traceback.print_exc()
        return None


def _process_sample_proprioception(task):
    """Helper function for parallel processing of proprioception samples for stats computation.

    Args:
        task: Tuple of (file_path, traj_idx, action_move_group_names, action_spec)

    Returns:
        numpy array of proprioception or None if error
    """
    file_path, traj_idx, action_move_group_names, action_spec = task

    try:
        with h5py.File(file_path, "r") as file:
            traj_key = f"traj_{traj_idx}"
            agent_data = file[traj_key]["obs"]["agent"]
            qpos_data = agent_data["qpos"]

            decoded_qpos = []
            for i in range(qpos_data.shape[0]):
                byte_array = qpos_data[i]
                json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                trajectory_dict = json.loads(json_string)
                decoded_qpos.append(trajectory_dict)

            all_proprioception = []
            for i in range(len(decoded_qpos)):
                qpos_vec = []
                for move_group in action_move_group_names:
                    qpos_dict = decoded_qpos[i]
                    if move_group not in qpos_dict:
                        qpos_vec.append(np.zeros(action_spec[move_group]))
                    elif move_group == "base":
                        qpos_vec.append(np.zeros(action_spec[move_group]))
                    elif move_group == "torso":
                        qpos_vec.append(np.array([qpos_dict[move_group][1]]))
                    else:
                        qpos_vec.append(np.array(qpos_dict[move_group]))
                qpos_vec = np.concatenate(qpos_vec)
                all_proprioception.append(qpos_vec)

            if len(all_proprioception) == 0:
                return None

            return np.array(all_proprioception, dtype=np.float32)

    except Exception as e:
        import traceback

        logger.debug(
            f"Error processing proprioception {file_path}, traj {traj_idx}: {e}"
        )
        traceback.print_exc()
        return None


class SpocDataset:
    def __init__(
        self,
        data_path: Union[str, List[str]],
        camera_names: list[str],
        action_move_group_names: list[str],
        action_spec: dict,
        action_keys: dict,
        action_chunk_size: int = 8,
        use_done_action: bool = False,
        trajectory_cache_file: Optional[Path] = None,
        use_proprioception: bool = False,
        input_sensors: Optional[List[str]] = None,
        phase_upsample_dict: dict[int, float] = {},
        point_camera_key: Optional[str] = None,
        randomize_prompts: bool = True,
        prompt_templates: dict[str, list[list[str]]] | None = None,
        prompt_sampling_prob_threshold: float = 0.15,
        prompt_sampling_temperature: float = 4.0,
        prompt_sampling_randomize_casing: bool = True,
        prompt_sampling_randomize_punctuation: bool = True,
    ):
        # Support both single path and multiple paths
        if isinstance(data_path, str):
            self.data_paths = [Path(data_path)]
        else:
            self.data_paths = [Path(p) for p in data_path]

        # For backward compatibility, keep data_path as the first path
        self.data_path = self.data_paths[0]

        # Store mapping from file to which data_path it belongs to
        # Initialize before _get_traj_files() is called
        self._file_to_data_path = {}

        # Input Params
        self.camera_names = camera_names
        self.input_window_size = 1
        self.randomize_prompts = randomize_prompts
        self.prompt_templates = (
            (prompt_templates or DEFAULT_PROMPT_TEMPLATES)
            if randomize_prompts
            else None
        )
        self.prompt_sampling_prob_threshold = prompt_sampling_prob_threshold
        self.prompt_sampling_temperature = prompt_sampling_temperature
        self.prompt_sampling_randomize_casing = prompt_sampling_randomize_casing
        self.prompt_sampling_randomize_punctuation = (
            prompt_sampling_randomize_punctuation
        )

        # Output Params
        self.action_move_group_names = action_move_group_names
        self.action_spec = action_spec
        self.action_dim = sum(
            self.action_spec[mg] for mg in self.action_move_group_names
        )
        self.action_chunk_size = action_chunk_size
        self.action_chunk_duration_ms = self.action_chunk_size * 100.0
        self.use_done_action = use_done_action
        self.action_keys = action_keys

        self.use_proprioception = use_proprioception

        self.input_sensors = input_sensors
        self.point_camera_key = point_camera_key

        # Internal bookkeeping
        self._files = self._get_traj_files()
        self.traj_idx_to_file_and_traj = {}
        self.traj_idx_to_length = {}
        self.traj_indices = []
        self.traj_lengths = []
        self.traj_cumsum_lengths = None
        self.phase_upsample_dict = phase_upsample_dict
        self._phase_samples = {}  # dict[str, list[(global_traj_idx, step)]]
        self._extra_phase_samples = []
        self.traj_idx_to_task_type = {}
        self._build_trajectory_bookkeeping(cache_file=trajectory_cache_file)

        # Decord
        decord.bridge.set_bridge("torch")

    @contextmanager
    def _open_video(self, video_path):
        """Context manager for VideoReader to ensure cleanup."""
        vr = VideoReader(video_path, ctx=cpu(0))
        try:
            yield vr
        finally:
            del vr

    def __len__(self):
        if self.traj_cumsum_lengths is None or len(self.traj_cumsum_lengths) == 0:
            return 0
        return self.traj_cumsum_lengths[-1] + len(self._extra_phase_samples)

    def __getitem__(self, idx):
        base_len = (
            self.traj_cumsum_lengths[-1] if len(self.traj_cumsum_lengths) > 0 else 0
        )
        if idx >= base_len:
            global_traj_idx, step = self._extra_phase_samples[idx - base_len]
        else:
            global_traj_idx, step = self._flat_idx_to_traj_idx(idx)
        file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
        file_path = self._files[file_idx]

        sample = {}
        # Open file handle once for all operations
        try:
            with h5py.File(file_path, "r") as file:
                obs_scene = self._parse_obs_scene(file, traj_idx)
                if "pickup_obj_image_points" in self.input_sensors:
                    sample.update(self._get_obj_image_points(file, traj_idx, obs_scene))
                    sample.update(
                        self._get_first_target_frame_repeated(file, file_path, traj_idx)
                    )

                sample.update(self._get_rgb_frames(file, file_path, traj_idx, step))
                sample.update(self._get_actions(file, traj_idx, step, global_traj_idx))
                sample.update(self._get_goal(file, traj_idx, obs_scene))
                sample.update(self._get_task_type(file, traj_idx, obs_scene))
                if self.use_proprioception:
                    sample.update(
                        self._get_proprioception(file, traj_idx, step, global_traj_idx)
                    )
        except OSError as e:
            logger.error(f"Failed to open corrupted file {file_path}: {e}")
            raise RuntimeError(f"Cannot read from corrupted file {file_path}") from e

        return sample

    def _parse_obs_scene(self, file: h5py.File, traj_idx: int) -> dict:
        """Parse obs_scene from h5 file. Helper method to avoid redundant parsing."""
        traj_key = f"traj_{traj_idx}"

        # Check if trajectory exists in the file
        if traj_key not in file:
            logger.debug(f"Trajectory '{traj_key}' not found in HDF5 file")
            raise ValueError(f"Trajectory '{traj_key}' not found in HDF5 file")

        try:
            obs_scene_data = file[traj_key]["obs_scene"]
            obs_scene_str = obs_scene_data.astype("T")[()]
            return json.loads(obs_scene_str)
        except KeyError as e:
            raise ValueError(
                f"Key 'obs_scene' not found in trajectory '{traj_key}'"
            ) from e

    def get_action_normalization_stats(
        self,
        num_workers=None,
        use_quantiles=False,
        use_mean_std=False,
        num_std=3,
        lower_quantile=0.01,
        upper_quantile=0.99,
        max_samples=10000,
    ):
        """Compute normalization statistics for actions across all valid trajectories.

        Args:
            num_workers: Number of parallel workers. If None, uses cpu_count().
            use_quantiles: If True, use quantile-based normalization instead of min-max.
            lower_quantile: Lower quantile to use (default: 0.01 for 1st percentile).
            upper_quantile: Upper quantile to use (default: 0.99 for 99th percentile).
            max_samples: Maximum number of action samples to collect for quantile computation.
        """
        if use_quantiles:
            return self._get_quantile_normalization_stats(
                lower_quantile, upper_quantile, max_samples, num_workers
            )
        if use_mean_std:
            return self._get_mean_std_normalization_stats(num_std=3)
        if num_workers is None:
            num_workers = cpu_count()

        logger.info(
            f"Computing action normalization stats with {num_workers} workers..."
        )

        # Prepare tasks for parallel processing
        tasks = []
        for global_traj_idx in self.traj_indices:
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]
            tasks.append(
                (file_path, traj_idx, self.action_keys, self.action_move_group_names)
            )

        # Process in parallel
        with Pool(num_workers) as pool:
            results = pool.map(_process_trajectory_stats, tasks)

        # Collect all mins and maxs for each move group
        all_mins_by_group = {mg: [] for mg in self.action_move_group_names}
        all_maxs_by_group = {mg: [] for mg in self.action_move_group_names}

        for result in results:
            if result is not None:
                mins_dict, maxs_dict = result
                for move_group in self.action_move_group_names:
                    if move_group in mins_dict:
                        all_mins_by_group[move_group].append(mins_dict[move_group])
                        all_maxs_by_group[move_group].append(maxs_dict[move_group])

        # Aggregate mins and maxs across all trajectories for each move group
        aggregated_mins = []
        aggregated_maxs = []

        for move_group in self.action_move_group_names:
            if len(all_mins_by_group[move_group]) > 0:
                # Take the minimum of all mins and maximum of all maxs
                group_mins = np.array(all_mins_by_group[move_group])
                group_maxs = np.array(all_maxs_by_group[move_group])

                aggregated_mins.append(np.min(group_mins, axis=0))
                aggregated_maxs.append(np.max(group_maxs, axis=0))
            else:
                # No data for this move group, use zeros
                logger.warning(
                    f"No data found for move group '{move_group}', using zeros"
                )
                aggregated_mins.append(np.zeros(self.action_spec[move_group]))
                aggregated_maxs.append(np.zeros(self.action_spec[move_group]))

        # Concatenate all move groups into single arrays
        final_mins = np.concatenate(aggregated_mins)
        final_maxs = np.concatenate(aggregated_maxs)

        # Convert to torch tensors
        final_mins = torch.from_numpy(final_mins).float()
        final_maxs = torch.from_numpy(final_maxs).float()

        logger.info(f"Computed normalization stats from {len(results)} trajectories")

        return final_mins, final_maxs

    def get_proprioception_normalization_stats(
        self,
        num_workers=None,
        use_quantiles=False,
        use_mean_std=False,
        num_std=3,
        lower_quantile=0.01,
        upper_quantile=0.99,
        max_samples=10000,
    ):
        """Compute normalization statistics for proprioception across all valid trajectories.

        Args:
            num_workers: Number of parallel workers. If None, uses cpu_count().
            use_quantiles: If True, use quantile-based normalization.
            use_mean_std: If True, use mean ± num_std*std normalization.
            num_std: Number of standard deviations for mean_std mode (default: 3).
            lower_quantile: Lower quantile (default: 0.01).
            upper_quantile: Upper quantile (default: 0.99).
            max_samples: Maximum number of trajectories to sample for computation.
        """
        if num_workers is None:
            num_workers = cpu_count()

        logger.info(
            f"Computing proprioception normalization stats with {num_workers} workers..."
        )

        # Sample from base dataset only (exclude grasp extra samples)
        base_len = (
            int(self.traj_cumsum_lengths[-1])
            if len(self.traj_cumsum_lengths) > 0
            else 0
        )
        num_samples = min(max_samples, base_len)
        sampled_indices = random.sample(range(base_len), num_samples)

        logger.info(
            f"Sampling {num_samples} proprioception trajectories from dataset of size {base_len}"
        )

        # Prepare tasks for parallel processing
        tasks = []
        for idx in sampled_indices:
            global_traj_idx, _ = self._flat_idx_to_traj_idx(idx)
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]
            tasks.append(
                (file_path, traj_idx, self.action_move_group_names, self.action_spec)
            )

        # Process in parallel
        logger.info(f"Processing {len(tasks)} samples in parallel...")
        with Pool(num_workers) as pool:
            results = pool.map(_process_sample_proprioception, tasks)

        # Collect all valid proprioception arrays
        all_proprioception = []
        for result in results:
            if result is not None:
                all_proprioception.append(result)

        if len(all_proprioception) == 0:
            raise ValueError(
                "No proprioception data could be collected for stats computation"
            )

        # Concatenate: (num_samples * timesteps, proprioception_dim)
        all_proprioception = np.concatenate(all_proprioception, axis=0)
        logger.info(
            f"Collected {all_proprioception.shape[0]} proprioception timesteps with dimension {all_proprioception.shape[1]}"
        )

        all_proprioception_torch = torch.from_numpy(all_proprioception).float()

        if use_mean_std:
            means = torch.mean(all_proprioception_torch, dim=0)
            stds = torch.std(all_proprioception_torch, dim=0)
            lower_bounds = means - num_std * stds
            upper_bounds = means + num_std * stds
            logger.info(f"Computed mean ± {num_std}*std proprioception stats")
            logger.info(f"Lower bounds: {lower_bounds}")
            logger.info(f"Upper bounds: {upper_bounds}")
            return lower_bounds, upper_bounds
        elif use_quantiles:
            lower_quantiles = torch.quantile(
                all_proprioception_torch, lower_quantile, dim=0
            )
            upper_quantiles = torch.quantile(
                all_proprioception_torch, upper_quantile, dim=0
            )
            logger.info(
                f"Computed quantile proprioception stats (q={lower_quantile}, q={upper_quantile})"
            )
            logger.info(f"Lower quantiles: {lower_quantiles}")
            logger.info(f"Upper quantiles: {upper_quantiles}")
            return lower_quantiles, upper_quantiles
        else:
            mins = torch.min(all_proprioception_torch, dim=0)[0]
            maxs = torch.max(all_proprioception_torch, dim=0)[0]
            logger.info(f"Computed min/max proprioception stats")
            logger.info(f"Mins: {mins}")
            logger.info(f"Maxs: {maxs}")
            return mins, maxs

    def _get_quantile_normalization_stats(
        self,
        lower_quantile=0.01,
        upper_quantile=0.99,
        max_samples=10000,
        num_workers=None,
    ):
        """Compute quantile-based normalization statistics by sampling actions from the dataset.

        Args:
            lower_quantile: Lower quantile to use (default: 0.01 for 1st percentile).
            upper_quantile: Upper quantile to use (default: 0.99 for 99th percentile).
            max_samples: Maximum number of samples to collect for quantile computation.
            num_workers: Number of parallel workers. If None, uses cpu_count().

        Returns:
            Tuple of (lower_quantile_values, upper_quantile_values) as torch tensors
        """
        if num_workers is None:
            num_workers = cpu_count()

        logger.info(
            f"Computing quantile normalization stats (q={lower_quantile}, q={upper_quantile}) with {num_workers} workers..."
        )

        # Sample from base dataset only (exclude grasp extra samples which
        # are duplicates and would cause IndexError in _flat_idx_to_traj_idx)
        base_len = (
            int(self.traj_cumsum_lengths[-1])
            if len(self.traj_cumsum_lengths) > 0
            else 0
        )
        num_samples = min(max_samples, base_len)
        sampled_indices = random.sample(range(base_len), num_samples)

        logger.info(f"Sampling {num_samples} actions from dataset of size {base_len}")

        # Prepare tasks for parallel processing
        tasks = []
        for idx in sampled_indices:
            global_traj_idx, _ = self._flat_idx_to_traj_idx(idx)
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]

            tasks.append(
                (
                    file_path,
                    traj_idx,
                    self.action_move_group_names,
                    self.action_spec,
                    self.action_keys,
                )
            )

        # Process in parallel
        logger.info(f"Processing {len(tasks)} samples in parallel...")
        with Pool(num_workers) as pool:
            results = pool.map(_process_sample_actions, tasks)

        # Collect all valid action arrays
        all_actions = []
        for result in results:
            if result is not None:
                all_actions.append(result)

        if len(all_actions) == 0:
            raise ValueError("No actions could be collected for quantile computation")

        # Concatenate all actions: (num_samples * timesteps, action_dim)
        all_actions = np.concatenate(all_actions, axis=0)
        logger.info(
            f"Collected {all_actions.shape[0]} action timesteps with dimension {all_actions.shape[1]}"
        )

        # Convert to torch and compute quantiles per dimension
        all_actions_torch = torch.from_numpy(all_actions).float()
        lower_quantiles = torch.quantile(all_actions_torch, lower_quantile, dim=0)
        upper_quantiles = torch.quantile(all_actions_torch, upper_quantile, dim=0)

        logger.info(
            f"Computed quantile normalization stats from {len(results)} samples"
        )
        logger.info(f"Lower quantile ({lower_quantile}): {lower_quantiles}")
        logger.info(f"Upper quantile ({upper_quantile}): {upper_quantiles}")

        return lower_quantiles, upper_quantiles

    def get_quantile_bin_edges(
        self,
        num_bins: int,
        normalization_mins: torch.Tensor,
        normalization_maxs: torch.Tensor,
        max_samples: int = 10000,
        num_workers: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute quantile-based bin edges for discretizing continuous actions.

        This method samples actions from the dataset, normalizes them to [-1, 1] using
        the provided normalization stats, and computes quantile-based bin edges for
        each action dimension independently.

        Args:
            num_bins: Number of bins per dimension
            normalization_mins: Tensor of minimum values for normalization (shape: action_dim)
            normalization_maxs: Tensor of maximum values for normalization (shape: action_dim)
            max_samples: Maximum number of action timesteps to sample (default: 10000)
            num_workers: Number of parallel workers. If None, uses cpu_count().

        Returns:
            Tensor of bin edges with shape (action_dim, num_bins + 1)
        """
        if num_workers is None:
            num_workers = cpu_count()

        logger.info(
            f"Computing quantile-based bin edges for {num_bins} bins with {num_workers} workers..."
        )

        # Sample from base dataset only (exclude grasp extra samples which
        # are duplicates and would cause IndexError in _flat_idx_to_traj_idx)
        base_len = (
            int(self.traj_cumsum_lengths[-1])
            if len(self.traj_cumsum_lengths) > 0
            else 0
        )
        num_samples = min(max_samples, base_len)
        sampled_indices = random.sample(range(base_len), num_samples)

        logger.info(f"Sampling {num_samples} actions from dataset of size {base_len}")

        # Prepare tasks for parallel processing
        tasks = []
        for idx in sampled_indices:
            global_traj_idx, _ = self._flat_idx_to_traj_idx(idx)
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]

            tasks.append(
                (
                    file_path,
                    traj_idx,
                    self.action_move_group_names,
                    self.action_spec,
                    self.action_keys,
                )
            )

        # Process in parallel to collect action samples
        logger.info(f"Processing {len(tasks)} samples in parallel...")
        with Pool(num_workers) as pool:
            results = pool.map(_process_sample_actions, tasks)

        # Collect all valid action arrays
        all_actions = []
        for result in results:
            if result is not None:
                all_actions.append(result)

        if len(all_actions) == 0:
            raise ValueError("No actions could be collected for quantile-based binning")

        # Concatenate all actions: (num_samples * timesteps, action_dim)
        all_actions = np.concatenate(all_actions, axis=0)
        logger.info(
            f"Collected {all_actions.shape[0]} action timesteps with dimension {all_actions.shape[1]}"
        )

        # Convert to torch
        all_actions_torch = torch.from_numpy(all_actions).float()

        # Normalize actions to [-1, 1] using the provided normalization stats
        mins = normalization_mins.view(1, -1)
        maxs = normalization_maxs.view(1, -1)
        all_actions_normalized = (
            2 * (all_actions_torch - mins) / (maxs - mins + 1e-8) - 1
        )
        all_actions_normalized = torch.clamp(all_actions_normalized, -1, 1)

        # Compute quantile-based bin edges for each dimension
        # We want num_bins + 1 edge points to create num_bins bins
        quantile_levels = torch.linspace(0, 1, num_bins + 1)

        # Compute bin edges per dimension: (num_bins + 1, action_dim)
        bin_edges_per_dim = []
        for dim in range(self.action_dim):
            dim_actions = all_actions_normalized[:, dim]
            dim_edges = torch.quantile(dim_actions, quantile_levels)
            bin_edges_per_dim.append(dim_edges)

        # Stack to get (action_dim, num_bins + 1)
        bin_edges_per_dim = torch.stack(bin_edges_per_dim, dim=0)

        logger.info(
            f"Computed quantile-based bin edges for {self.action_dim} dimensions"
        )
        logger.info(f"Bin edges shape: {bin_edges_per_dim.shape}")
        logger.info(
            f"Example bin edges for dim 0: {bin_edges_per_dim[0, :5]}...{bin_edges_per_dim[0, -5:]}"
        )

        return bin_edges_per_dim

    def _save_trajectory_cache(self, cache_file):
        """Save trajectory bookkeeping to cache file."""
        import pickle

        try:
            cache_data = {
                "traj_idx_to_file_and_traj": self.traj_idx_to_file_and_traj,
                "traj_idx_to_length": self.traj_idx_to_length,
                "traj_indices": self.traj_indices,
                "traj_lengths": self.traj_lengths,
                "traj_cumsum_lengths": self.traj_cumsum_lengths,
                "files": self._files,
                "phase_samples": self._phase_samples,
                "traj_idx_to_task_type": self.traj_idx_to_task_type,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved trajectory index cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save trajectory cache: {e}")

    def get_task_types_for_samples(self, indices) -> list[str]:
        """Get task_types for multiple sample indices (vectorized)."""
        if len(indices) == 0:
            return []

        indices = np.asarray(indices)
        base_len = (
            self.traj_cumsum_lengths[-1] if len(self.traj_cumsum_lengths) > 0 else 0
        )

        # Vectorized conversion: split indices into regular and extra phase samples
        is_extra_phase = indices >= base_len
        regular_mask = ~is_extra_phase

        # Get global trajectory indices for regular samples using vectorized searchsorted
        global_traj_indices = np.empty(len(indices), dtype=int)

        if np.any(regular_mask):
            regular_indices = indices[regular_mask]
            # Vectorized searchsorted for all regular indices at once
            traj_idxs = np.searchsorted(
                self.traj_cumsum_lengths, regular_indices, side="right"
            )
            # Get global trajectory indices
            global_traj_indices[regular_mask] = np.array(self.traj_indices)[traj_idxs]

        if np.any(is_extra_phase):
            extra_phase_indices = indices[is_extra_phase] - base_len
            extra_phase_global_traj_indices = np.array(
                [self._extra_phase_samples[idx][0] for idx in extra_phase_indices]
            )
            global_traj_indices[is_extra_phase] = extra_phase_global_traj_indices

        # Batch lookup task types
        task_types = [
            self.traj_idx_to_task_type.get(global_traj_idx, "unknown")
            for global_traj_idx in global_traj_indices
        ]
        return task_types

    def _get_task_type(
        self, file: h5py.File, traj_idx: int, obs_scene: dict
    ) -> dict[str, str]:
        """Get task_type for a trajectory.
        Returns:
            Dictionary with:
                - task_type: str - The task type (e.g., 'pick_and_place', 'pick', etc.)
        """
        sample = {}
        try:
            task_type: str = obs_scene["task_type"]
            sample["task_type"] = task_type
        except KeyError as e:
            raise RuntimeError(f"Key 'task_type' not found in obs_scene: {e}") from e
        return sample

    def _build_trajectory_bookkeeping_from_scratch(self):
        global_traj_idx = 0

        # Build a mapping from (data_path, relative_path) to file path to handle
        # cases where the same relative path exists in multiple data paths
        path_to_file = {}
        for file_path in self._files:
            # Find which data_path this file belongs to
            data_path = self._file_to_data_path[file_path]
            data_root = data_path.parent.parent.parent
            relative_path_str = str(file_path.relative_to(data_root))
            path_to_file[(data_root, relative_path_str)] = file_path

        # Build list of files to process in order from all data paths
        files_to_process = []

        # Process each data path's valid_trajectory_index.json
        # valid_trajectory_index.json lives at data_root level (data_path.parent.parent.parent)
        # and its relative paths are also relative to data_root.
        seen_data_roots = set()
        for data_path in self.data_paths:
            data_root = data_path.parent.parent.parent
            if data_root in seen_data_roots:
                continue
            seen_data_roots.add(data_root)

            valid_traj_index_path = data_root / "valid_trajectory_index.json"
            if not valid_traj_index_path.exists():
                logger.warning(
                    f"valid_trajectory_index.json not found at {valid_traj_index_path}, skipping this data path"
                )
                continue

            logger.info(f"Loading valid_trajectory_index.json from {data_root}")
            with open(valid_traj_index_path, "r") as f:
                valid_traj_index = json.load(f)

            # Build list of files to process from this data path
            for house_data in valid_traj_index.values():
                for relative_path_str, traj_dict in house_data.items():
                    key = (data_root, relative_path_str)
                    if key in path_to_file:
                        file_path = path_to_file[key]
                        files_to_process.append(
                            (file_path, traj_dict, self._file_to_data_path[file_path])
                        )
                    else:
                        logger.warning(
                            f"File {relative_path_str} in valid_trajectory_index.json not found in data directory under {data_root}"
                        )

        if len(files_to_process) == 0:
            raise ValueError("No valid trajectories found in any data path")

        file_idx_to_task_types = {}
        logger.info("Pre-loading task types from h5 files...")

        tasks = [(file_path, traj_dict) for file_path, traj_dict, _ in files_to_process]

        # Load task types in parallel
        num_workers = cpu_count()
        with Pool(num_workers) as pool:
            results = pool.map(_load_task_types_from_file, tasks)

        # Map results back to file indices and track data paths
        for file_idx, (file_path, traj_dict, data_path) in enumerate(files_to_process):
            file_idx_to_task_types[file_idx] = results[file_idx]

        # Process each file and trajectory
        for file_idx, (file_path, traj_dict, data_path) in tqdm(
            enumerate(files_to_process),
            total=len(files_to_process),
            desc="Building trajectory index",
        ):
            for traj_key, traj_length_from_json in traj_dict.items():
                traj_idx = int(traj_key.split("_")[1])

                # Use length from JSON, subtract 1 (don't include first action)
                traj_length = traj_length_from_json - 1
                # Subtract one more if we're not using done action
                traj_length = (
                    traj_length - 1 if not self.use_done_action else traj_length
                )

                if traj_length > 0:
                    self.traj_idx_to_file_and_traj[global_traj_idx] = (
                        file_idx,
                        traj_idx,
                    )
                    self.traj_idx_to_length[global_traj_idx] = traj_length
                    self.traj_indices.append(global_traj_idx)
                    self.traj_lengths.append(traj_length)
                    task_type = file_idx_to_task_types.get(file_idx, {}).get(
                        traj_idx, "unknown"
                    )
                    self.traj_idx_to_task_type[global_traj_idx] = task_type
                    global_traj_idx += 1

        # Update self._files to only contain the files we actually processed
        self._files = [file_path for file_path, _, _ in files_to_process]
        logger.info(
            f"Built bookkeeping for {len(self._files)} files and {global_traj_idx} trajectories from {len(self.data_paths)} data paths"
        )

        if len(self.traj_lengths) > 0:
            self.traj_cumsum_lengths = np.cumsum(self.traj_lengths)
        else:
            self.traj_cumsum_lengths = np.array([])

        # Scan for phase samples
        if self.phase_upsample_dict:
            self._phase_samples = self._scan_phase_samples()
        else:
            self._phase_samples = {}

    def _scan_phase_samples(self, num_workers=None):
        """Scan all trajectories for samples whose action chunks contain any phase in phase_upsample_dict.

        Returns:
            Dict mapping each phase (str) to a list of (global_traj_idx, step) tuples.
        """
        if num_workers is None:
            num_workers = cpu_count()

        target_str_phases = list(self.phase_upsample_dict.keys())
        logger.info(f"Scanning for phase samples (phases={target_str_phases})...")

        tasks = []
        for global_traj_idx in self.traj_indices:
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]
            traj_length = self.traj_idx_to_length[global_traj_idx]
            tasks.append(
                (
                    file_path,
                    traj_idx,
                    traj_length,
                    self.action_chunk_size,
                    target_str_phases,
                )
            )

        with Pool(num_workers) as pool:
            results = pool.map(_find_samples_with_phases, tasks)

        phase_samples = {p: [] for p in target_str_phases}
        for global_traj_idx, result in zip(self.traj_indices, results):
            for str_phase, steps in result.items():
                for step in steps:
                    phase_samples[str_phase].append((global_traj_idx, step))

        for phase, samples in phase_samples.items():
            logger.info(
                f"Found {len(samples)} samples for phase {phase} across {len(self.traj_indices)} trajectories"
            )
        return phase_samples

    def _get_mean_std_normalization_stats(self, num_std=3, num_workers=None):
        """Compute mean ± num_std*std normalization statistics for actions across all valid trajectories.

        Args:
            num_std: Number of standard deviations to use for the range (default: 3).
            num_workers: Number of parallel workers. If None, uses cpu_count().

        Returns:
            Tuple of (lower_bounds, upper_bounds) as torch tensors, where:
                lower_bounds = mean - num_std * std
                upper_bounds = mean + num_std * std
        """
        if num_workers is None:
            num_workers = cpu_count()

        logger.info(
            f"Computing mean ± {num_std}*std normalization stats with {num_workers} workers..."
        )

        # Prepare tasks for parallel processing
        tasks = []
        for global_traj_idx in self.traj_indices:
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            file_path = self._files[file_idx]
            tasks.append(
                (file_path, traj_idx, self.action_keys, self.action_move_group_names)
            )

        # Process in parallel
        with Pool(num_workers) as pool:
            results = pool.map(_process_trajectory_mean_std_stats, tasks)

        # Collect all means and stds for each move group
        all_means_by_group = {mg: [] for mg in self.action_move_group_names}
        all_stds_by_group = {mg: [] for mg in self.action_move_group_names}

        for result in results:
            if result is not None:
                means_dict, stds_dict = result
                for move_group in self.action_move_group_names:
                    if move_group in means_dict:
                        all_means_by_group[move_group].append(means_dict[move_group])
                        all_stds_by_group[move_group].append(stds_dict[move_group])

        # Aggregate means and stds across all trajectories for each move group
        aggregated_means = []
        aggregated_stds = []

        for move_group in self.action_move_group_names:
            if len(all_means_by_group[move_group]) > 0:
                # Average the means and stds across all trajectories
                group_means = np.array(all_means_by_group[move_group])
                group_stds = np.array(all_stds_by_group[move_group])

                aggregated_means.append(np.mean(group_means, axis=0))
                aggregated_stds.append(np.mean(group_stds, axis=0))
            else:
                raise ValueError(f"Move group {move_group} not found in stats")

        # Concatenate all move groups into single arrays
        final_means = np.concatenate(aggregated_means)
        final_stds = np.concatenate(aggregated_stds)

        # Compute lower and upper bounds: mean ± num_std * std
        lower_bounds = final_means - num_std * final_stds
        upper_bounds = final_means + num_std * final_stds

        # Convert to torch tensors
        lower_bounds = torch.from_numpy(lower_bounds).float()
        upper_bounds = torch.from_numpy(upper_bounds).float()

        logger.info(
            f"Computed mean ± {num_std}*std normalization stats from {len(results)} trajectories"
        )
        logger.info(f"Lower bounds (mean - {num_std}*std): {lower_bounds}")
        logger.info(f"Upper bounds (mean + {num_std}*std): {upper_bounds}")

        return lower_bounds, upper_bounds

    def _get_traj_files(self):
        files = []
        for data_path in self.data_paths:
            for house_dir in os.listdir(data_path):
                if os.path.isdir(data_path / house_dir):
                    for file in os.listdir(data_path / house_dir):
                        if file.endswith("h5"):
                            file_path = data_path / house_dir / file
                            files.append(file_path)
                            # Store which data_path this file belongs to
                            self._file_to_data_path[file_path] = data_path
        return sorted(files)  # Sort for consistent ordering

    def _build_trajectory_bookkeeping(self, cache_file: Optional[Path] = None):
        """Build trajectory bookkeeping.

        Args:
            cache_file: Optional path to cache file. If provided and exists, loads from cache.
                       If provided and doesn't exist, saves to cache after building.
        """
        import pickle

        # Check which data paths have valid_trajectory_index.json (at data_root level)
        paths_with_index = [
            dp
            for dp in self.data_paths
            if (dp.parent.parent.parent / "valid_trajectory_index.json").exists()
        ]
        paths_without_index = [
            dp for dp in self.data_paths if dp not in paths_with_index
        ]
        if paths_without_index and paths_with_index:
            missing = ", ".join(str(p) for p in paths_without_index)
            raise FileNotFoundError(
                f"The following data paths are missing valid_trajectory_index.json and need postprocessing: {missing}"
            )
        if cache_file is not None and cache_file.exists():
            # Try to load from cache first if cache_file is provided
            logger.info(f"Loading trajectory index from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.traj_idx_to_file_and_traj = cached_data[
                        "traj_idx_to_file_and_traj"
                    ]
                    self.traj_idx_to_length = cached_data["traj_idx_to_length"]
                    self.traj_indices = cached_data["traj_indices"]
                    self.traj_lengths = cached_data["traj_lengths"]
                    self.traj_cumsum_lengths = cached_data["traj_cumsum_lengths"]
                    self._files = cached_data["files"]
                    self._phase_samples = cached_data.get("phase_samples", {})
                    self.traj_idx_to_task_type = cached_data["traj_idx_to_task_type"]
                logger.info(f"Loaded {len(self.traj_indices)} trajectories from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache, rebuilding: {e}")
                self._build_trajectory_bookkeeping_from_scratch()
        else:
            # Build from scratch
            logger.info(
                "Building trajectory index from scratch (this may take a while)..."
            )
            self._build_trajectory_bookkeeping_from_scratch()

        # Compute phase upsampling from phase_samples dict
        if self.phase_upsample_dict and self._phase_samples:
            extra_samples = []
            for phase, upsample_pct in self.phase_upsample_dict.items():
                phase_samples = self._phase_samples.get(phase, [])
                if not phase_samples:
                    continue
                num_full_copies = int(upsample_pct / 100)
                remaining_frac = (upsample_pct / 100) - num_full_copies
                for _ in range(num_full_copies):
                    extra_samples.extend(phase_samples)
                if remaining_frac > 0:
                    num_fractional = round(len(phase_samples) * remaining_frac)
                    extra_samples.extend(phase_samples[:num_fractional])
                logger.info(
                    f"Phase {phase} upsampling ({upsample_pct}%): {len(phase_samples)} samples"
                )

            self._extra_phase_samples = extra_samples
            logger.info(
                f"Total phase upsampling: added {len(extra_samples)} extra samples (new total: {len(self)})"
            )

    def _flat_idx_to_traj_idx(self, flat_idx):
        if len(self.traj_cumsum_lengths) == 0:
            raise IndexError(f"Dataset is empty, cannot access index {flat_idx}")

        traj_idx = np.searchsorted(self.traj_cumsum_lengths, flat_idx, side="right")

        if traj_idx > 0:
            step = flat_idx - self.traj_cumsum_lengths[traj_idx - 1]
        else:
            step = flat_idx

        global_traj_idx = self.traj_indices[traj_idx]
        return global_traj_idx, step

    def _get_file_and_traj_idx(self, global_traj_idx) -> tuple[int, int]:
        if global_traj_idx not in self.traj_idx_to_file_and_traj:
            raise ValueError(f"Global trajectory index {global_traj_idx} not found")

        file_idx, traj_idx = self.traj_idx_to_file_and_traj[global_traj_idx]
        return file_idx, traj_idx

    def _decode_dict_data(self, traj_idx, keys, data):

        # Load and decode data if not cached (all action types are under traj_i/actions/)
        all_dict_data = {}
        for key in keys:
            key_data = data[key]
            trajectories = []
            for i in range(key_data.shape[0]):
                byte_array = key_data[i]
                json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                trajectory_dict = json.loads(json_string)
                trajectories.append(trajectory_dict)
            all_dict_data[key] = trajectories

        return all_dict_data

    def _get_rgb_frames(
        self, file: h5py.File, file_path: Path, traj_idx: int, step: int
    ) -> dict[str, torch.Tensor]:
        sample = {}
        traj_key = f"traj_{traj_idx}"

        window_start = step - self.input_window_size + 1
        window_end = step + 1

        for camera_name in self.camera_names:
            try:
                # Get video path
                obs_data = file[traj_key]["obs"]["sensor_data"][camera_name]
                video_filename = obs_data[:].tobytes().decode("utf-8").rstrip("\x00")
                # Use the data_path that corresponds to this file
                data_path = self._file_to_data_path[file_path]
                # file_path.parent is the directory containing the h5 file, relative to data_path
                # Get the relative path from data_path to file_path.parent
                relative_dir = file_path.parent.relative_to(data_path)
                video_path = str(data_path / relative_dir / video_filename)

                # Open video once and read all needed frames
                with self._open_video(video_path) as vr:
                    frames = []
                    for i in range(window_start, window_end):
                        if i < 0:
                            # Get shape from first frame for padding
                            H, W, C = vr[0].shape
                            frames.append(torch.zeros(H, W, C))
                        else:
                            frame = vr[i]  # Get single frame
                            frames.append(frame)

                    # Stack frames: (input_window_size, H, W, C)
                    sample[camera_name] = torch.stack(frames)
            except KeyError as e:
                logger.warning(
                    f"Camera '{camera_name}' not found in {file_path} trajectory {traj_idx}: {e}. Skipping this camera."
                )
                continue

        return sample

    def _get_actions(
        self, file: h5py.File, traj_idx: int, step: int, global_traj_idx: int
    ) -> dict[str, torch.Tensor]:
        """Return 1 action chunk per window size timestep.

        Returns actions of shape (input_window_size, action_chunk_size, action_dim)
        where each timestep in the window gets one action chunk.
        """
        try:
            action_data = file["traj_" + str(traj_idx)]["actions"]
        except KeyError as e:
            logger.error(f"Trajectory 'traj_{traj_idx}' not found: {e}")
            raise RuntimeError(f"Trajectory 'traj_{traj_idx}' not found") from e

        # Get trajectory length from bookkeeping
        traj_length = self.traj_idx_to_length[global_traj_idx]

        # Get unique action keys needed
        unique_action_keys = list(set(self.action_keys.values()))

        # Decode all required action keys
        all_decoded_data = self._decode_dict_data(
            traj_idx, unique_action_keys, action_data
        )

        window_start = step - self.input_window_size + 1
        window_end = step + 1

        # Collect one action chunk per timestep in the window
        all_chunks = []
        all_is_pad = []

        for window_timestep in range(window_start, window_end):
            # For each timestep in the window, get one action chunk
            # Actions start from window_timestep + 1 (actions are 1-indexed)
            chunk_start = window_timestep + 1
            chunk_end = window_timestep + 1 + self.action_chunk_size

            traj_chunk_start = max(1, chunk_start)
            # Use traj_length + 1 because actions are 1-indexed
            traj_chunk_end = min(traj_length + 1, chunk_end)

            chunk_actions = []
            for i in range(traj_chunk_start, traj_chunk_end):
                action_vec = []
                for move_group in self.action_move_group_names:
                    # Get the appropriate action key for this move group
                    action_key = self.action_keys[move_group]
                    decoded_data = all_decoded_data[action_key]

                    # If move group is not in the decoded data, use zeros
                    if move_group in decoded_data[i]:
                        action_vec.append(
                            decoded_data[i][move_group][: self.action_spec[move_group]]
                        )
                    else:
                        if self.use_done_action and decoded_data[i] == {}:
                            raise NotImplementedError("Need to handle done action")
                        elif move_group == "torso":
                            action_vec.append(np.zeros(self.action_spec[move_group]))
                        else:
                            raise ValueError(
                                f"Move group {move_group} not found in data"
                            )
                action_vec = np.concatenate(action_vec)
                chunk_actions.append(action_vec)

            # Pad chunk if needed
            if len(chunk_actions) == 0:
                chunk_array = np.zeros((0, self.action_dim), dtype=np.float32)
            else:
                chunk_array = np.array(chunk_actions, dtype=np.float32)

            padded_chunk, chunk_is_pad = pad_data(
                chunk_array, chunk_start, chunk_end, traj_chunk_start, traj_chunk_end
            )
            all_chunks.append(padded_chunk)
            all_is_pad.append(chunk_is_pad)

        # Stack all chunks: (input_window_size, action_chunk_size, action_dim)
        actions_tensor = torch.stack([torch.from_numpy(chunk) for chunk in all_chunks])
        # pad_data returns is_pad as torch.Tensor, so no need to convert from numpy
        is_pad_tensor = torch.stack(all_is_pad)

        return {"actions": actions_tensor, "actions_is_pad": is_pad_tensor}

    def _get_proprioception(
        self, file: h5py.File, traj_idx: int, step: int, global_traj_idx: int
    ) -> dict[str, torch.Tensor]:
        """Return absolute joint positions as proprioception input.

        Returns:
            Dictionary with:
                - proprioception: (input_window_size, action_dim) tensor of absolute joint positions
        """
        proprioception = []
        traj_key = f"traj_{traj_idx}"

        try:
            agent_data = file[traj_key]["obs"]["agent"]
        except KeyError as e:
            logger.error(
                f"Agent observation data not found in trajectory '{traj_key}': {e}"
            )
            raise RuntimeError(
                f"Agent observation data not found in trajectory '{traj_key}'"
            ) from e

        # Decode qpos data
        qpos_data = agent_data["qpos"]
        decoded_qpos = []
        for i in range(qpos_data.shape[0]):
            byte_array = qpos_data[i]
            json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
            trajectory_dict = json.loads(json_string)
            decoded_qpos.append(trajectory_dict)

        # Get trajectory length from bookkeeping
        traj_length = self.traj_idx_to_length[global_traj_idx]

        window_start = step - self.input_window_size + 1
        window_end = step + 1

        num_grippers = sum(1 for mg in self.action_move_group_names if "gripper" in mg)

        for i in range(window_start, window_end):
            if i < 0:
                # Pad with zeros for timesteps before the trajectory starts
                proprioception.append(torch.zeros(self.action_dim + num_grippers))
            else:
                # Make sure we don't access beyond valid trajectory length
                if i >= len(decoded_qpos):
                    proprioception.append(torch.zeros(self.action_dim + num_grippers))
                else:
                    # Extract absolute joint positions for all action move groups
                    qpos_tensors = []
                    for move_group in self.action_move_group_names:
                        qpos_dict = decoded_qpos[i]
                        if move_group not in qpos_dict:
                            logger.warning(
                                f"Move group {move_group} not found in qpos. Padding with zeros."
                            )
                            qpos_tensors.append(
                                torch.zeros(self.action_spec[move_group])
                            )
                        elif move_group == "base":
                            qpos_tensors.append(
                                torch.zeros(self.action_spec[move_group])
                            )
                        elif move_group == "torso":
                            qpos_tensors.append(
                                torch.tensor([qpos_dict[move_group][1]])
                            )
                        else:
                            qpos_tensors.append(torch.tensor(qpos_dict[move_group]))
                    proprioception.append(torch.cat(qpos_tensors))

        # Stack into (input_window_size, action_dim) tensor
        return {"proprioception": torch.stack(proprioception)}

    def _get_first_target_frame_repeated(
        self, file: h5py.File, file_path: Path, traj_idx: int
    ) -> dict[str, torch.Tensor]:
        """Get the first head_camera frame of the trajectory, repeated across the input window.

        Returns:
            Dictionary with:
                - head_camera_first_frame: (input_window_size, H, W, C) tensor
        """
        traj_key = f"traj_{traj_idx}"
        camera_name = next(
            (
                cam
                for cam in [
                    "exo_camera_1",
                    "droid_shoulder_light_randomization",
                    "head_camera",
                ]
                if cam in self.input_sensors
            ),
            "head_camera",
        )
        obs_data = file[traj_key]["obs"]["sensor_data"][camera_name]
        video_filename = obs_data[:].tobytes().decode("utf-8").rstrip("\x00")
        data_path = self._file_to_data_path[file_path]
        relative_dir = file_path.parent.relative_to(data_path)
        video_path = str(data_path / relative_dir / video_filename)

        with self._open_video(video_path) as vr:
            first_frame = vr[0]  # (H, W, C)

        repeated = first_frame.unsqueeze(0).repeat(
            self.input_window_size, 1, 1, 1
        )  # (input_window_size, H, W, C)
        return {"first_target_frame_repeated": repeated}

    def _get_goal(
        self, file: h5py.File, traj_idx: int, obs_scene: dict
    ) -> dict[str, str]:
        sample = {}
        try:
            # Check if this is a point-conditioned task
            is_point_conditioned = "pickup_obj_image_points" in self.input_sensors

            if not self.randomize_prompts:
                goal: str = obs_scene["task_description"]
            else:
                task_type: str = obs_scene["task_type"]

                if is_point_conditioned:
                    # For point-conditioned tasks, use point-conditioned templates (no object names needed)
                    # Map task_type to its point-conditioned version
                    assert self.prompt_templates is not None
                    point_task_type = f"{task_type}_with_point"

                    prompt_template_group = random.choice(
                        self.prompt_templates[point_task_type]
                    )
                    prompt_template = random.choice(prompt_template_group)
                    goal = prompt_template  # No formatting needed for point-conditioned templates
                else:
                    # For regular tasks, sample referral expressions and use regular templates
                    referral_expressions: dict[str, list[tuple[str, float]]] = (
                        obs_scene["referral_expressions"]
                    )
                    sampled_referral_exps: dict[str, str] = {}
                    for obj_name in referral_expressions.keys():
                        exps = [
                            exp
                            for exp, prob in referral_expressions[obj_name]
                            if prob > self.prompt_sampling_prob_threshold
                        ]
                        # if there are no high-probability expressions, they're all the same-ish so use all of them
                        if len(exps) == 0:
                            exps = [exp for exp, _ in referral_expressions[obj_name]]
                        # if there aren't any expressions, return the default task description
                        if len(exps) == 0:
                            goal: str = obs_scene["task_description"]
                            break
                        # softmax sample with bias towards shorter expressions
                        probs = [
                            np.exp(-len(exp.split()) / self.prompt_sampling_temperature)
                            for exp in exps
                        ]
                        probs = np.array(probs) / np.sum(probs)
                        idx = np.random.choice(len(exps), p=probs)
                        sampled_referral_exps[obj_name] = exps[idx]
                    else:
                        assert self.prompt_templates is not None
                        prompt_template_group = random.choice(
                            self.prompt_templates[task_type]
                        )
                        prompt_template = random.choice(prompt_template_group)
                        prompt = prompt_template.format(**sampled_referral_exps)
                        assert "{" not in prompt and "}" not in prompt, (
                            f"Badly formatted prompt: {prompt}"
                        )
                        goal = prompt

            if self.prompt_sampling_randomize_casing and random.random() < 0.5:
                goal = goal.lower()
            if self.prompt_sampling_randomize_punctuation and random.random() < 0.5:
                goal = goal.replace(".", "").replace("?", "").replace("!", "")

            sample["goal"] = goal
        except KeyError as e:
            logger.error(f"Key not found in obs_scene: {e}")
            raise RuntimeError(f"Key not found in obs_scene: {e}") from e
        return sample

    def _extract_points_from_json_format(
        self, object_image_points, obj_key: str, camera_name: str
    ) -> list:
        """Extract points list from Jan 27 JSON format."""
        json_string = object_image_points[0].tobytes().decode("utf-8").rstrip("\x00")
        points_dict = json.loads(json_string)
        return points_dict[obj_key][camera_name]

    def _extract_points_from_hdf5_format(
        self, object_image_points, obj_key: str, camera_name: str
    ) -> list:
        """Extract points list from Feb 10/12 HDF5 format."""
        points_dict = object_image_points[obj_key][camera_name]
        num_points = points_dict["num_points"][0]
        return points_dict["points"][0][:num_points]

    def _extract_points_from_direct_format(
        self, object_image_points, obj_key: str, camera_name: str
    ) -> list:
        """Extract points list from direct points dataset format."""
        points_dataset = object_image_points[obj_key][camera_name]["points"]
        first_frame_points = points_dataset[0]  # numpy array, shape (10, 2)
        valid_mask = ~np.any(np.isnan(first_frame_points), axis=1)
        valid_points = first_frame_points[valid_mask]
        return valid_points.tolist() if len(valid_points) > 0 else []

    def _get_points_list(
        self, object_image_points, obj_key: str, camera_name: str
    ) -> list:
        """Get points list trying different data formats in order."""
        # Try Jan 27 JSON format
        try:
            return self._extract_points_from_json_format(
                object_image_points, obj_key, camera_name
            )
        except:
            pass

        # Try Feb 10/12 HDF5 format
        try:
            return self._extract_points_from_hdf5_format(
                object_image_points, obj_key, camera_name
            )
        except:
            pass

        # Try direct points dataset format
        try:
            return self._extract_points_from_direct_format(
                object_image_points, obj_key, camera_name
            )
        except:
            pass

    def _get_obj_image_points(
        self, file: h5py.File, traj_idx: int, obs_scene: dict
    ) -> dict[str, torch.Tensor]:
        """Get object_image_points from file[traj_key]["obs"]["extra"]["object_image_points"].

        Only gets the point at the beginning of the trajectory (step 0) and repeats it
        for all timesteps in the window.

        Handles three task types:
        - Pick tasks: returns 1 point (x, y) -> padded to shape (1, 4)
        - Pick-and-place tasks: returns 2 points (x1, y1, x2, y2) -> shape (1, 4)
          where (x1, y1) is the pickup object point and (x2, y2) is the receptacle point.
        - Opening tasks: returns 1 point (x, y) from door_handle -> padded to shape (1, 4)

        Returns:
            Dictionary with:
                - pickup_obj_image_points: (1, 4) tensor of object image points
        """
        traj_key = f"traj_{traj_idx}"
        object_image_points = file[traj_key]["obs"]["extra"]["object_image_points"]
        camera_name = self.point_camera_key
        task_type = obs_scene.get("task_type", "").lower()

        # Determine object key from task type
        if task_type == "door_open":
            obj_key = "door_handle"
        elif "pick" in task_type or "open" in task_type:
            obj_key = "pickup_obj"
        else:
            raise KeyError(
                f"Cannot determine object key for task_type='{task_type}' in '{traj_key}'"
            )
        # Get pickup object point
        pickup_points_list = self._get_points_list(
            object_image_points, obj_key, camera_name
        )
        selected_point = random.choice(pickup_points_list)
        pickup_point = [selected_point[0], selected_point[1]]

        # Get receptacle point for pick-and-place tasks
        if "place" in task_type:
            receptacle_points_list = self._get_points_list(
                object_image_points, "place_receptacle", camera_name
            )
            if len(receptacle_points_list) == 0:
                raise ValueError(
                    f"Empty receptacle points list for pick-and-place task in trajectory '{traj_key}'"
                )
            selected_receptacle = random.choice(receptacle_points_list)
            receptacle_point = [selected_receptacle[0], selected_receptacle[1]]
        else:
            receptacle_point = [0.0, 0.0]  # Pad for consistency

        # Combine points: always 4 coordinates [x1, y1, x2, y2]
        points_list = pickup_point + receptacle_point

        # Convert to tensor and repeat across input_window_size
        points_tensor = torch.tensor(points_list, dtype=torch.float32)  # Shape: (4,)
        points_tensor = points_tensor.unsqueeze(0).repeat(
            self.input_window_size, 1
        )  # Shape: (input_window_size, 4)

        return {"pickup_obj_image_points": points_tensor}
