"""
SynthManip Dataset Wrapper for MolmoBot Training

Loads robot demonstration data from h5 files with video observations stored as mp4 files.
Compatible with the MolmoBot training pipeline.

Sampling Behavior:
    The dataset is indexed by trajectory (not by timestep). Each call to get()
    selects a trajectory uniformly, then samples a timestep within that trajectory
    according to the weight function (or uniformly if no weight function is set).

    This means longer trajectories are NOT oversampled — each trajectory has equal
    probability of being selected regardless of length.

Expected directory structure:
    {data_path}/{split}/house_*/*.h5

    Where:
    - data_path: Task type directory (e.g., /path/to/SomeTaskConfig)
                 Must contain train/ and val/ subdirectories
    - split: "train" or "val"
    - house_*: Per-house directories containing h5 files

H5 file format:
    - traj_{i}/obs/sensor_data/{camera_name}: path to mp4 video
    - traj_{i}/actions/{action_key}: JSON-encoded or native action data
    - traj_{i}/obs_scene: JSON with task_description
    - traj_{i}/success: trajectory success flags
    - valid_traj_mask: boolean mask for valid trajectories
    - stats/traj_{i}/{action_key}: per-trajectory normalization statistics

    Object image points format (HDF5 native groups):
        traj_{i}/obs/extra/object_image_points/
        ├── {object_name}/
        │   ├── {camera_name}/
        │   │   ├── points           # (T, max_points, 2) - NaN-padded normalized coords
        │   │   └── num_points       # (T, 1) - valid point count per frame

    Action frame indexing:
    - Frame 0 is padding (may contain empty dict {})
    - Frames 1 to N are actual action commands
    - Frame N+1 is the "done" action (excluded when use_done_action=False)
    - After adjustment: traj_length = number of valid timesteps = N
    - Valid action frame indices are [1, traj_length] inclusive
    - Valid steps are [0, traj_length - 1]
"""

import json
import logging
import random
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import decord
import h5py
import numpy as np
from decord import VideoReader, cpu as decord_cpu

from olmo.data.dataset import Dataset
from olmo.data.image_warping_utils import apply_fisheye_warping, warp_point_coordinates
from olmo.data.robot_processing import RobotPreprocessor

# ── Sibling modules ────────────────────────────────────────────────────────────
from olmo.data.synthmanip_config import (
    DEFAULT_PROMPT_TEMPLATES,
    SynthmanipDatasetConfig,
    synthmanip_config_registry,
)
from olmo.data.synthmanip_grasp_sampling import compute_grasp_aware_weights
from olmo.data.synthmanip_stats import (
    compute_action_normalization_stats,
    compute_state_normalization_stats,
)
from olmo.data.synthmanip_utils import (
    ZED2_CAMERAS,
    _decode_h5_string,
    _open_h5_with_retry,
    _pad_action_chunk,
    compute_camera_distances,
)

log = logging.getLogger(__name__)


# Type alias for step weight functions (kept for any external code that references it)
StepWeightFunction = Callable[[h5py.File, int, int, List[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SynthmanipDataset(Dataset):
    """Dataset wrapper for SynthManip-format robot demonstrations.

    Indexing:
        ``len(dataset)`` returns trajectory count (not timestep count).
        Each ``get(idx)`` call selects trajectory *idx*, then samples a
        timestep within it according to the configured weight function
        (or uniformly when no weight function is configured).

        Per-step weights are computed lazily and cached on first access.

    Returned example keys:
        image, question, answers, style, state, action, action_is_pad, metadata
        (plus object_image_points_* and policy_phase* when enabled)
    """

    def __init__(self, config: SynthmanipDatasetConfig):
        super().__init__()
        self.config = config
        self.data_path = self._resolve_data_path(Path(config.data_path), config.split)
        self.camera_names = config.camera_names
        self.action_move_group_names = config.action_move_group_names
        self.action_spec = config.action_spec
        self.action_keys = config.action_keys
        self.input_window_size = config.input_window_size
        self.action_horizon = config.action_horizon
        self.use_done_action = config.use_done_action
        self.style = config.style

        self.action_dim = sum(self.action_spec[mg] for mg in self.action_move_group_names)
        self.state_spec = config.state_spec if config.state_spec is not None else self.action_spec
        self.state_indices = config.state_indices if config.state_indices is not None else {}
        self.state_dim = sum(self.state_spec[mg] for mg in self.action_move_group_names)

        self.robot_preprocessor: Optional[RobotPreprocessor] = None
        if config.robot_processor_config is not None:
            self.robot_preprocessor = config.robot_processor_config.build_preprocessor()
            log.info("SynthmanipDataset: Using robot preprocessor for action/state normalization")

        log.info(
            f"SynthmanipDataset config:\n"
            f"  action_move_group_names: {self.action_move_group_names}\n"
            f"  action_spec:             {self.action_spec}\n"
            f"  action_keys:             {self.action_keys}\n"
            f"  action_dim:              {self.action_dim}"
        )

        self._files: List[Path] = []
        self.traj_idx_to_file_and_traj: Dict[int, Tuple[int, int]] = {}
        self.traj_idx_to_length: Dict[int, int] = {}
        self.traj_indices: List[int] = []
        self.traj_lengths: List[int] = []

        # Lazy caches for per-step weights
        self._step_weights: Dict[int, np.ndarray] = {}
        self._step_cumsum: Dict[int, np.ndarray] = {}

        self._build_trajectory_bookkeeping()
        decord.bridge.set_bridge("torch")

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _resolve_data_path(self, data_path: Path, split: str) -> Path:
        """Resolve *data_path* to the directory that contains house_* subdirs."""
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        if not data_path.is_dir():
            raise ValueError(f"Data path is not a directory: {data_path}")

        split_path = data_path / split
        if not split_path.exists():
            raise ValueError(
                f"Split directory does not exist: {split_path}\n"
                f"Expected structure: {data_path}/{split}/house_*/*.h5"
            )
        if not split_path.is_dir():
            raise ValueError(f"Split path is not a directory: {split_path}")

        log.info(f"Using data path: {split_path}")
        return split_path

    def __len__(self) -> int:
        return len(self.traj_indices)

    # ── Weighted sampling ──────────────────────────────────────────────────────

    def _ensure_weights_cached(self, global_traj_idx: int) -> None:
        """Lazily compute and cache per-step weights for *global_traj_idx*."""
        if global_traj_idx in self._step_weights:
            return

        traj_length = self.traj_idx_to_length[global_traj_idx]

        if self.config.weighted_sampling and self.config.weight_config:
            file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
            try:
                with _open_h5_with_retry(self._files[file_idx]) as f:
                    weights = compute_grasp_aware_weights(f, traj_idx, traj_length, self.config.weight_config)
            except Exception as e:
                log.warning(f"Weight computation failed for traj {global_traj_idx}: {e}, using uniform")
                weights = np.ones(traj_length, dtype=np.float32)
        else:
            weights = np.ones(traj_length, dtype=np.float32)

        weights = np.maximum(weights, 1e-8)
        cumsum = np.cumsum(weights)
        cumsum /= cumsum[-1]

        self._step_weights[global_traj_idx] = weights
        self._step_cumsum[global_traj_idx] = cumsum

    def _sample_step_weighted(self, global_traj_idx: int, rng: np.random.Generator) -> int:
        """Inverse-CDF sample a timestep within *global_traj_idx*."""
        self._ensure_weights_cached(global_traj_idx)
        cumsum = self._step_cumsum[global_traj_idx]
        step = int(np.searchsorted(cumsum, rng.random()))
        return min(step, len(cumsum) - 1)

    # ── Camera selection helpers ───────────────────────────────────────────────

    def _select_zed_camera_by_distance(
        self,
        traj_group: h5py.Group,
        camera_names: List[str],
        select_furthest: bool,
    ) -> List[str]:
        """Replace ZED2 placeholder cameras with the chosen ZED2 analogue camera."""
        if select_furthest:
            try:
                distances = compute_camera_distances(traj_group, ZED2_CAMERAS, timestep=0)
                selected_cam = max(distances, key=distances.get)
                log.debug(f"ZED selection: {selected_cam} (furthest, distances={distances})")
            except (KeyError, IndexError) as e:
                log.debug(f"Could not compute distances, using default ZED camera: {e}")
                selected_cam = ZED2_CAMERAS[0]
        else:
            selected_cam = ZED2_CAMERAS[0]

        result, zed_replaced = [], False
        for cam in camera_names:
            if cam in ZED2_CAMERAS:
                if not zed_replaced:
                    result.append(selected_cam)
                    zed_replaced = True
            else:
                result.append(cam)
        return result

    def _select_effective_cameras(
        self, rng: np.random.Generator, traj_group: h5py.Group
    ) -> List[str]:
        """Return the effective camera list for this sample (ZED selection applied)."""
        effective = list(self.camera_names)

        if self.config.max_exo_views == 2:
            effective = ZED2_CAMERAS[:]
            for cam in self.camera_names:
                if cam not in effective:
                    effective.append(cam)
        elif self.config.furthest_camera_prob > 0:
            select_furthest = rng.random() < self.config.furthest_camera_prob
            effective = self._select_zed_camera_by_distance(traj_group, effective, select_furthest)

        return effective

    # ── Conditioning frame selection ───────────────────────────────────────────

    def _select_conditioning_frame_idx(
        self,
        f: h5py.File,
        rng: np.random.Generator,
        traj_idx: int,
        traj_length: int,
    ) -> int:
        """Pick a valid conditioning frame index according to config."""
        if isinstance(self.config.conditioning_frame, int):
            requested = min(self.config.conditioning_frame, traj_length - 1)
            valid_frames, required_pairs = self._get_valid_conditioning_frames(f, traj_idx, requested)
            if requested not in valid_frames:
                raise ValueError(
                    f"Requested conditioning frame {requested} is not valid. "
                    f"Required (object, camera) pairs: {required_pairs}. "
                    f"Valid frames in [0, {requested}]: {valid_frames}"
                )
            return requested

        if self.config.conditioning_frame == "random_first_10":
            max_cond = min(9, traj_length - 1)
            valid_frames, required_pairs = self._get_valid_conditioning_frames(f, traj_idx, max_cond)
            if not valid_frames:
                raise ValueError(
                    f"No valid conditioning frames in [0, {max_cond}]. "
                    f"Required (object, camera) pairs: {required_pairs}"
                )
            return int(rng.choice(valid_frames))

        raise ValueError(
            f"Invalid conditioning_frame value: {self.config.conditioning_frame!r}. "
            f"Expected int or 'random_first_10'."
        )

    # ── Point prompt formatting ────────────────────────────────────────────────

    def _format_point_prompt(
        self,
        goal: str,
        object_image_points: Dict[str, Dict[str, np.ndarray]],
    ) -> str:
        """Append Molmo html-v2 ``<points>`` tags to *goal*.

        Formats all objects with visible points on ``config.point_prompt_camera`` as::

            <points coords="1 xxx yyy 2 xxx yyy ...">object_name</points>

        Coordinates are 0–1000 integers (Molmo html-v2 scale).
        """
        cam = self.config.point_prompt_camera
        point_tags = []
        for obj_name, cam_dict in object_image_points.items():
            if "gripper" in obj_name:
                continue
            pts = cam_dict.get(cam)
            if pts is None or len(pts) == 0:
                continue
            coords_parts = [
                f"{i + 1} {int(np.clip(pt[0], 0.0, 1.0) * 1000):03d} {int(np.clip(pt[1], 0.0, 1.0) * 1000):03d}"
                for i, pt in enumerate(pts)
            ]
            coords_str = "1 " + " ".join(coords_parts)
            point_tags.append(f'<points coords="{coords_str}">{obj_name}</points>')

        return (goal + " " + " ".join(point_tags)) if point_tags else goal

    # ── Main data-loading entry point ──────────────────────────────────────────

    def get(self, idx: int, rng: np.random.Generator) -> Dict[str, Any]:
        """Return one training example for MolmoBot.

        *idx* selects a trajectory; the timestep within it is sampled
        according to the configured weight function.
        """
        global_traj_idx = self.traj_indices[idx]
        step = self._sample_step_weighted(global_traj_idx, rng)
        file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
        file_path = self._files[file_idx]

        log.debug(f"get: idx={idx}, file={file_path.name}, traj={traj_idx}, step={step}")

        try:
            with _open_h5_with_retry(file_path) as f:
                traj_key = f"traj_{traj_idx}"
                traj_group = f[traj_key]

                effective_cameras = self._select_effective_cameras(rng, traj_group)

                # Conditioning frame + object image points
                conditioning_frame_idx = None
                object_image_points_conditioning = None
                object_image_points_current = None
                conditioning_image = None
                extra_frame_indices = None

                if self.config.load_object_image_points:
                    traj_length = self.traj_idx_to_length[global_traj_idx]
                    conditioning_frame_idx = self._select_conditioning_frame_idx(
                        f, rng, traj_idx, traj_length
                    )
                    extra_frame_indices = [conditioning_frame_idx]

                # Load video frames
                frames, extra_frames = self._get_camera_frames(
                    f, file_path, traj_idx, step, extra_frame_indices,
                    camera_names_override=effective_cameras,
                )

                if self.config.load_object_image_points and conditioning_frame_idx is not None:
                    conditioning_image = extra_frames.get(conditioning_frame_idx, [])
                    object_image_points_conditioning, object_image_points_current = (
                        self._get_object_image_points(f, traj_idx, conditioning_frame_idx, step)
                    )
                    # Transform point coordinates for fisheye-warped cameras
                    if self.config.cameras_to_warp and hasattr(self, '_orig_frame_shapes'):
                        for cam in self.config.cameras_to_warp:
                            if cam not in self._orig_frame_shapes:
                                continue
                            orig_h, orig_w = self._orig_frame_shapes[cam]
                            for pts_dict in [object_image_points_conditioning, object_image_points_current]:
                                if not pts_dict:
                                    continue
                                for obj_name in pts_dict:
                                    if cam in pts_dict[obj_name] and len(pts_dict[obj_name][cam]) > 0:
                                        pts_dict[obj_name][cam] = warp_point_coordinates(
                                            pts_dict[obj_name][cam], orig_h, orig_w
                                        )

                actions, action_is_pad = self._get_actions(f, traj_idx, step, global_traj_idx)
                goal = self._get_goal(f, traj_idx)

                if self.config.use_point_prompts and object_image_points_conditioning:
                    goal = self._format_point_prompt(goal, object_image_points_conditioning)

                state = self._get_state(f, traj_idx, step, global_traj_idx)

                policy_phase = policy_phase_name = None
                if self.config.load_policy_phase:
                    policy_phase, policy_phase_name = self._get_policy_phase(f, traj_idx, step)

        except OSError as e:
            raise RuntimeError(f"Cannot read from file {file_path}") from e

        # Append conditioning frame image for point-prompt camera
        image_list = list(frames)
        if self.config.use_point_prompts:
            if not conditioning_image:
                raise RuntimeError(
                    "use_point_prompts=True but no conditioning image was loaded. "
                    "Ensure load_object_image_points=True and conditioning_frame is set."
                )
            cam_idx = self.camera_names.index(self.config.point_prompt_camera)
            if cam_idx >= len(conditioning_image):
                raise RuntimeError(
                    f"point_prompt_camera '{self.config.point_prompt_camera}' index {cam_idx} "
                    f"out of range for conditioning_image (len={len(conditioning_image)})"
                )
            image_list.append(conditioning_image[cam_idx])

        # State shape guard
        actual_state_dim = state.shape[0] if hasattr(state, 'shape') else len(state)
        if actual_state_dim != self.state_dim:
            if self.action_spec == {'arm': 7, 'gripper': 1}:
                state = state[:(self.action_spec['arm'] + self.config.gripper_representation_count)]
            else:
                raise ValueError(
                    f"State shape {state.shape} != expected {self.state_dim} "
                    f"for action_spec {self.action_spec}"
                )

        # Normalize
        repo_id = "synthmanip"
        if self.robot_preprocessor is not None:
            if state is not None:
                state = self.robot_preprocessor.normalize_state(state, repo_id)
            if actions is not None:
                actions = self.robot_preprocessor.normalize_action(actions, repo_id)

        result: Dict[str, Any] = {
            "image": image_list,
            "question": goal,
            "answers": "",
            "style": self.style,
            "state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "metadata": {
                "traj_index": global_traj_idx,
                "step": step,
                "file_path": str(file_path),
                "traj_idx": traj_idx,
                "split": self.config.split,
                "repo_id": repo_id,
            },
        }

        if self.config.load_object_image_points:
            result["object_image_points_conditioning"] = object_image_points_conditioning
            result["object_image_points_current"] = object_image_points_current
            result["conditioning_image"] = conditioning_image
            result["conditioning_frame_idx"] = conditioning_frame_idx

        if self.config.load_policy_phase:
            result["policy_phase"] = policy_phase
            result["policy_phase_name"] = policy_phase_name

        return result

    # ── Trajectory indexing ────────────────────────────────────────────────────

    def _build_trajectory_bookkeeping(self) -> None:
        """Load ``valid_trajectory_index.json`` and build internal index structures."""
        json_index_path = self.data_path / "valid_trajectory_index.json"
        if not json_index_path.exists():
            raise FileNotFoundError(
                f"Required index file not found: {json_index_path}\n"
                f"Each split directory must contain valid_trajectory_index.json."
            )
        self._load_from_json_index(json_index_path)

    def _load_from_json_index(self, json_path: Path) -> None:
        """Parse ``valid_trajectory_index.json`` into internal bookkeeping dicts.

        JSON format::

            {
                "house_123": {
                    "house_123/trajectories_batch_1.h5": {
                        "traj_0": 103,
                        "traj_2": 88,
                        ...
                    },
                    ...
                },
                ...
            }
        """
        log.info(f"Loading trajectory index from {json_path}")

        with open(json_path) as fh:
            index_data = json.load(fh)

        # Collect and sort h5 relative paths for deterministic ordering
        h5_rel_paths = sorted({
            h5_rel for house_data in index_data.values() for h5_rel in house_data
        })
        self._files = [self.data_path / rel for rel in h5_rel_paths]
        file_path_to_idx = {rel: idx for idx, rel in enumerate(h5_rel_paths)}
        log.info(f"Found {len(self._files)} h5 files from JSON index")

        global_traj_idx = 0
        for house_data in index_data.values():
            for h5_rel, traj_data in house_data.items():
                file_idx = file_path_to_idx[h5_rel]
                for traj_key, traj_length in traj_data.items():
                    traj_idx = int(traj_key.split("_")[1])
                    # Frame 0 = padding; last frame = done action (excluded unless use_done_action)
                    traj_length -= 1 if self.use_done_action else 2
                    if traj_length > 0:
                        self.traj_idx_to_file_and_traj[global_traj_idx] = (file_idx, traj_idx)
                        self.traj_idx_to_length[global_traj_idx] = traj_length
                        self.traj_indices.append(global_traj_idx)
                        self.traj_lengths.append(traj_length)
                        global_traj_idx += 1

        self.traj_cumsum_lengths = (
            np.cumsum(self.traj_lengths) if self.traj_lengths else np.array([])
        )
        log.info(f"Loaded {global_traj_idx} valid trajectories from JSON index")

    def _get_file_and_traj_idx(self, global_traj_idx: int) -> Tuple[int, int]:
        if global_traj_idx not in self.traj_idx_to_file_and_traj:
            raise ValueError(f"Global trajectory index {global_traj_idx} not found")
        return self.traj_idx_to_file_and_traj[global_traj_idx]

    # ── Video frame loading ────────────────────────────────────────────────────

    @contextmanager
    def _open_video(self, video_path: str):
        """Context manager for decord VideoReader."""
        log.debug(f"Opening video: {video_path}")
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        try:
            yield vr
        finally:
            del vr

    def _get_camera_frames(
        self,
        f: h5py.File,
        file_path: Path,
        traj_idx: int,
        step: int,
        extra_frame_indices: Optional[List[int]] = None,
        camera_names_override: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]]]:
        """Load observation-window frames plus optional extra frames.

        Returns:
            (observation_frames, extra_frames) where extra_frames maps
            frame_index → list-of-frames-per-camera.
        """
        traj_key = f"traj_{traj_idx}"
        camera_names = camera_names_override if camera_names_override is not None else self.camera_names

        frame_indices = [
            step - (self.input_window_size - 1 - i) * self.config.obs_step_delta
            for i in range(self.input_window_size)
            if step - (self.input_window_size - 1 - i) * self.config.obs_step_delta >= 0
        ]

        observation_frames: List[np.ndarray] = []
        extra_frames: Dict[int, List[np.ndarray]] = {idx: [] for idx in (extra_frame_indices or [])}

        for camera_name in camera_names:
            try:
                obs_data = f[traj_key]["obs"]["sensor_data"][camera_name]
                video_filename = obs_data[:].tobytes().decode("utf-8").rstrip("\x00")
                video_path = str(self.data_path / file_path.parent.name / video_filename)

                with self._open_video(video_path) as vr:
                    for i in frame_indices:
                        if i < 0:
                            val = vr[0]
                            first = val.asnumpy() if hasattr(val, "asnumpy") else (val.numpy() if hasattr(val, "numpy") else np.asarray(val))
                            H, W, C = first.shape
                            frame = np.zeros((H, W, C), dtype=np.uint8)
                        else:
                            val = vr[i]
                            frame = val.asnumpy() if hasattr(val, "asnumpy") else (val.numpy() if hasattr(val, "numpy") else np.asarray(val))
                        if camera_name in self.config.cameras_to_warp:
                            if not hasattr(self, '_orig_frame_shapes'):
                                self._orig_frame_shapes: Dict[str, Tuple[int, int]] = {}
                            self._orig_frame_shapes[camera_name] = frame.shape[:2]
                            frame = apply_fisheye_warping(frame)
                        observation_frames.append(frame)

                    if extra_frame_indices:
                        for idx in extra_frame_indices:
                            val = vr[idx]
                            frame = val.asnumpy() if hasattr(val, "asnumpy") else (val.numpy() if hasattr(val, "numpy") else np.asarray(val))
                            if camera_name in self.config.cameras_to_warp:
                                frame = apply_fisheye_warping(frame)
                            extra_frames[idx].append(frame)

            except KeyError as e:
                log.warning(f"Camera '{camera_name}' not found in {file_path} traj {traj_idx}: {e}")

        return observation_frames, extra_frames

    # ── Action loading ─────────────────────────────────────────────────────────

    def _decode_action_data(
        self,
        action_data: h5py.Group,
        action_keys: List[str],
    ) -> Dict[str, Any]:
        """Decode action data from h5 (JSON-bytes or native numeric)."""
        decoded = {}
        for key in action_keys:
            if key not in action_data:
                continue
            dataset = action_data[key]
            is_json_bytes = (dataset.dtype == np.uint8 and len(dataset.shape) == 2)

            if not is_json_bytes and dataset.dtype.kind in ('f', 'i', 'u'):
                decoded[key] = dataset[:]
            else:
                trajectories = []
                try:
                    for i in range(dataset.shape[0]):
                        byte_array = dataset[i]
                        json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                        trajectories.append(json.loads(json_string))
                    decoded[key] = trajectories
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    decoded[key] = dataset[:]
        return decoded

    def _get_actions(
        self,
        f: h5py.File,
        traj_idx: int,
        step: int,
        global_traj_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get action chunk for the current step."""
        traj_key = f"traj_{traj_idx}"
        action_data = f[traj_key]["actions"]
        traj_length = self.traj_idx_to_length[global_traj_idx]

        unique_keys = list(set(self.action_keys.values()))
        if "joint_pos_rel" in unique_keys and "joint_pos" not in unique_keys and "joint_pos" in action_data:
            unique_keys.append("joint_pos")

        decoded = self._decode_action_data(action_data, unique_keys)

        # Action indices: 1-indexed (frame 0 = padding)
        chunk_start = step + 1
        chunk_end = step + 1 + self.action_horizon
        actual_start = max(1, chunk_start)
        actual_end = min(traj_length + 1, chunk_end)

        chunk_actions = []
        for i in range(actual_start, actual_end):
            action_vec = []
            for move_group in self.action_move_group_names:
                action_key = self.action_keys[move_group]

                if action_key not in decoded:
                    raise ValueError(f"Action key '{action_key}' not found. Available: {list(decoded.keys())}")
                if i >= len(decoded[action_key]):
                    raise IndexError(
                        f"Frame {i} out of bounds for {action_key} "
                        f"(len={len(decoded[action_key])}, step={step}, traj_length={traj_length})"
                    )

                frame_data = decoded[action_key][i]

                if isinstance(frame_data, dict) and not frame_data:
                    raise ValueError(f"Empty action data at frame {i} for {action_key}")

                if move_group in frame_data:
                    val = np.array(frame_data[move_group], dtype=np.float32)
                    action_vec.append(val[:self.action_spec[move_group]])
                elif action_key == "joint_pos_rel" and "joint_pos" in decoded:
                    if i >= len(decoded["joint_pos"]):
                        raise IndexError(f"Frame {i} out of bounds for joint_pos")
                    jp_frame = decoded["joint_pos"][i]
                    if isinstance(jp_frame, dict) and move_group in jp_frame:
                        val = np.array(jp_frame[move_group], dtype=np.float32)
                        action_vec.append(val[:self.action_spec[move_group]])
                    else:
                        raise ValueError(f"Move group '{move_group}' not found in joint_pos_rel or joint_pos")
                elif move_group == "torso":
                    action_vec.append(np.zeros(self.action_spec[move_group], dtype=np.float32))
                else:
                    raise ValueError(
                        f"Move group '{move_group}' not found in action data. "
                        f"Available keys: {list(frame_data.keys())}"
                    )

            chunk_actions.append(np.concatenate(action_vec))

        chunk_array = np.stack(chunk_actions) if chunk_actions else np.zeros((0, self.action_dim), dtype=np.float32)
        return _pad_action_chunk(chunk_array, chunk_start, chunk_end, actual_start, actual_end)

    # ── State loading ──────────────────────────────────────────────────────────

    def _get_state(
        self,
        f: h5py.File,
        traj_idx: int,
        step: int,
        global_traj_idx: int,
    ) -> np.ndarray:
        """Get robot joint state (qpos) for the current step."""
        traj_key = f"traj_{traj_idx}"
        try:
            qpos_data = f[traj_key]["obs"]["agent"]["qpos"]
            frame_idx = max(0, min(step, qpos_data.shape[0] - 1))

            if not (qpos_data.dtype == np.uint8 and len(qpos_data.shape) == 2):
                raise ValueError("qpos data is not JSON-encoded bytes")

            byte_array = qpos_data[frame_idx]
            qpos_dict = json.loads(byte_array.tobytes().decode("utf-8").rstrip("\x00"))

            state_vec = []
            for move_group in self.action_move_group_names:
                if move_group in qpos_dict:
                    val = qpos_dict[move_group]
                    if "gripper" in move_group:
                        val = val[:self.config.gripper_representation_count]
                    elif move_group in self.state_indices:
                        val = [val[i] for i in self.state_indices[move_group]]
                    state_vec.extend(val if isinstance(val, (list, tuple)) else [val])
                else:
                    state_vec.extend([0.0] * self.state_spec[move_group])

            return np.array(state_vec, dtype=np.float32)

        except KeyError:
            log.debug(f"obs/agent/qpos not found for traj {traj_idx}, using zeros")
            return np.zeros(self.state_dim, dtype=np.float32)

    # ── Goal / task description ────────────────────────────────────────────────

    def _get_goal(self, f: h5py.File, traj_idx: int) -> str:
        """Get task description (with optional prompt randomization)."""
        traj_key = f"traj_{traj_idx}"
        try:
            scene_data = json.loads(_decode_h5_string(f[traj_key]["obs_scene"]))

            if not self.config.randomize_prompts:
                ret: str = scene_data.get("task_description", "")
            else:
                ret = self._sample_randomized_prompt(scene_data)

            if self.config.prompt_sampling_randomize_casing and random.random() < 0.5:
                ret = ret.lower()
            if self.config.prompt_sampling_randomize_punctuation and random.random() < 0.5:
                ret = ret.replace(".", "")

            return ret
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f"Could not load goal for traj {traj_idx}: {e}")
            return ""

    def _sample_randomized_prompt(self, scene_data: Dict[str, Any]) -> str:
        """Sample a randomized prompt from referral expressions and template groups."""
        task_type: str = scene_data["task_type"]
        referral_expressions: dict = scene_data["referral_expressions"]
        sampled_refs: Dict[str, str] = {}

        for obj_name, exp_list in referral_expressions.items():
            exps = [exp for exp, prob in exp_list if prob > self.config.prompt_sampling_prob_threshold]
            if not exps:
                exps = [exp for exp, _ in exp_list]
            if not exps:
                return scene_data.get("task_description", "")
            probs = np.array([np.exp(-len(e.split()) / self.config.prompt_sampling_temperature) for e in exps])
            probs /= probs.sum()
            sampled_refs[obj_name] = exps[np.random.choice(len(exps), p=probs)]

        if task_type not in DEFAULT_PROMPT_TEMPLATES:
            log.warning(f"No prompt templates for task_type '{task_type}', using task_description")
            return scene_data.get("task_description", "")

        # Alias keys so templates can use any variant
        for a, b in [("pickup_obj_name", "pickup_name"), ("place_name", "place_receptacle")]:
            if a in sampled_refs and b not in sampled_refs:
                sampled_refs[b] = sampled_refs[a]
            elif b in sampled_refs and a not in sampled_refs:
                sampled_refs[a] = sampled_refs[b]

        prompt_template = random.choice(random.choice(DEFAULT_PROMPT_TEMPLATES[task_type]))
        prompt = prompt_template.format(**sampled_refs)
        assert "{" not in prompt and "}" not in prompt, f"Badly formatted prompt: {prompt}"
        return prompt

    # ── Policy phase ───────────────────────────────────────────────────────────

    def _get_policy_phase(
        self,
        f: h5py.File,
        traj_idx: int,
        step: int,
    ) -> Tuple[int, str]:
        """Return (phase_int, phase_name) for *step* in *traj_idx*."""
        traj_key = f"traj_{traj_idx}"
        try:
            phase_int = int(f[traj_key]["obs"]["extra"]["policy_phase"][step])
        except (KeyError, IndexError) as e:
            log.warning(f"Could not load policy_phase for traj {traj_idx} step {step}: {e}")
            return 0, "unknown"

        try:
            scene_data = json.loads(_decode_h5_string(f[traj_key]["obs_scene"]))
            int_to_name = {v: k for k, v in scene_data.get("policy_phases", {}).items()}
            phase_name = int_to_name.get(phase_int, "unknown")
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f"Could not load policy_phases mapping for traj {traj_idx}: {e}")
            phase_name = "unknown"

        return phase_int, phase_name

    # ── Object image points ────────────────────────────────────────────────────

    MIN_CONDITIONING_POINTS = 5

    def _decode_object_image_points_frame(
        self,
        f: h5py.File,
        traj_idx: int,
        frame_idx: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Decode object_image_points for *frame_idx* from HDF5 native format."""
        traj_key = f"traj_{traj_idx}"
        points_group = f[traj_key]["obs"]["extra"]["object_image_points"]
        result: Dict[str, Dict[str, np.ndarray]] = {}

        for obj_name in points_group:
            result[obj_name] = {}
            cam_group = points_group[obj_name]
            for camera_name in cam_group:
                cam_data = cam_group[camera_name]
                points = cam_data["points"][frame_idx]
                num_pts = min(
                    int(cam_data["num_points"][frame_idx, 0]),
                    self.config.max_points_in_conditioning_frame,
                )
                result[obj_name][camera_name] = points[:num_pts].astype(np.float32)

        return result

    def _get_required_conditioning_pairs(
        self, frame_0_points: Dict[str, Dict[str, np.ndarray]]
    ) -> set:
        """Return (object, camera) pairs that must be visible in every conditioning frame.

        Based on frame 0: all configured non-wrist cameras with ≥ MIN_CONDITIONING_POINTS.
        """
        configured_cameras = {cam for cam in self.camera_names if "wrist" not in cam.lower()}
        return {
            (obj_name, camera_name)
            for obj_name, camera_dict in frame_0_points.items()
            for camera_name, points in camera_dict.items()
            if camera_name in configured_cameras and len(points) >= self.MIN_CONDITIONING_POINTS
        }

    def _frame_is_valid_conditioning(
        self,
        frame_points: Dict[str, Dict[str, np.ndarray]],
        required_pairs: set,
    ) -> bool:
        """Return True if *frame_points* satisfies all *required_pairs*."""
        return all(
            len(frame_points[obj][cam]) >= self.MIN_CONDITIONING_POINTS
            for obj, cam in required_pairs
        )

    def _get_valid_conditioning_frames(
        self,
        f: h5py.File,
        traj_idx: int,
        max_frame: int,
    ) -> Tuple[List[int], set]:
        """Return (valid_frame_indices, required_pairs) for frames 0…max_frame."""
        frame_0_points = self._decode_object_image_points_frame(f, traj_idx, 0)
        required_pairs = self._get_required_conditioning_pairs(frame_0_points)

        if not required_pairs:
            return [0], required_pairs

        valid_frames = [0]  # Frame 0 is always valid by definition
        for frame_idx in range(1, max_frame + 1):
            frame_points = self._decode_object_image_points_frame(f, traj_idx, frame_idx)
            if self._frame_is_valid_conditioning(frame_points, required_pairs):
                valid_frames.append(frame_idx)

        return valid_frames, required_pairs

    def _get_object_image_points(
        self,
        f: h5py.File,
        traj_idx: int,
        conditioning_frame_idx: int,
        current_step: int,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """Return (conditioning_frame_points, current_frame_points)."""
        return (
            self._decode_object_image_points_frame(f, traj_idx, conditioning_frame_idx),
            self._decode_object_image_points_frame(f, traj_idx, current_step),
        )

    # ── Normalization stats (thin wrappers around synthmanip_stats) ───────────

    def get_action_normalization_stats(
        self,
        num_workers: Optional[int] = None,
        mode: str = "min_max",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute action normalization statistics.

        Args:
            num_workers: Parallel workers (defaults to cpu_count).
            mode: "min_max", "mean_std", or "quantile".
            **kwargs: For quantile mode: lower_quantile, upper_quantile, max_samples.

        Returns:
            (low, high) arrays of shape (action_dim,).
        """
        return compute_action_normalization_stats(self, num_workers=num_workers, mode=mode, **kwargs)

    def get_state_normalization_stats(
        self,
        num_workers: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute state min/max normalization statistics.

        Returns:
            (global_min, global_max) arrays of shape (state_dim,).
        """
        return compute_state_normalization_stats(self, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_synthmanip_dataset(
    dataset_name: str,
    split: str,
    **kwargs,
) -> SynthmanipDataset:
    """Build a SynthmanipDataset from the global config registry.

    The training script must call ``synthmanip_config_registry.register()``
    before invoking this function.

    Args:
        dataset_name: Registry key, e.g. ``"synthmanip/door_opening"``.
        split: ``"train"`` or ``"val"``.
        **kwargs: Override any SynthmanipDatasetConfig fields.

    Returns:
        SynthmanipDataset instance.
    """
    registered_config = synthmanip_config_registry.get(dataset_name)

    if registered_config is None:
        raise ValueError(
            f"No configuration registered for dataset '{dataset_name}'. "
            f"Register a SynthmanipDatasetConfig via synthmanip_config_registry.register() first.\n"
            f"Example:\n"
            f"  from olmo.data.synthmanip_config import synthmanip_config_registry, SynthmanipDatasetConfig\n"
            f"  config = SynthmanipDatasetConfig(\n"
            f"      data_path='/path/to/data/ConfigName',\n"
            f"      camera_names=[...],\n"
            f"      action_move_group_names=[...],\n"
            f"      action_spec={{...}},\n"
            f"      action_keys={{...}},\n"
            f"  )\n"
            f"  synthmanip_config_registry.register('{dataset_name}', config)"
        )

    config = replace(registered_config, split=split)
    if kwargs:
        config = SynthmanipDatasetConfig(**{**asdict(config), **kwargs})

    log.info(f"Building SynthManip dataset '{dataset_name}' split='{split}'")
    return SynthmanipDataset(config)
