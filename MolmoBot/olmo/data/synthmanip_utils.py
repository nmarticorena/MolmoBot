"""Shared utility functions for SynthManip dataset loading.

Low-level helpers for HDF5 I/O, camera geometry, and action padding
that are used by the dataset class and normalization-stats workers alike.
"""

import logging
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np

log = logging.getLogger(__name__)


# Default ZED2 camera names for distance-based selection
ZED2_CAMERAS = ["randomized_zed2_analogue_1", "randomized_zed2_analogue_2"]


# ---------------------------------------------------------------------------
# Camera / object geometry helpers
# ---------------------------------------------------------------------------

def get_camera_position(traj_group: h5py.Group, camera_name: str, timestep: int = 0) -> np.ndarray:
    """Extract camera world position from cam2world_gl transform.

    cam2world_gl is (T, 4, 4) where [:3, 3] is the translation (camera position in world).
    """
    cam2world = traj_group[f"obs/sensor_param/{camera_name}/cam2world_gl"][timestep]
    return cam2world[:3, 3]


def get_pickup_object_position(traj_group: h5py.Group, timestep: int = 0) -> np.ndarray:
    """Extract pickup object world position from obj_start.

    obj_start is (T, 7) with [x, y, z, qw, qx, qy, qz].
    """
    obj_pose = traj_group["obs/extra/obj_start"][timestep]
    return obj_pose[:3]


def compute_camera_distances(
    traj_group: h5py.Group,
    camera_names: List[str],
    timestep: int = 0,
) -> Dict[str, float]:
    """Compute distances from cameras to pickup object.

    Args:
        traj_group: HDF5 group for the trajectory
        camera_names: List of camera names to compute distances for
        timestep: Frame index for position extraction

    Returns:
        {cam_name: distance_float}
    """
    obj_pos = get_pickup_object_position(traj_group, timestep)
    return {
        cam_name: float(np.linalg.norm(get_camera_position(traj_group, cam_name, timestep) - obj_pos))
        for cam_name in camera_names
    }


# ---------------------------------------------------------------------------
# HDF5 I/O helpers
# ---------------------------------------------------------------------------

def _decode_h5_string(dataset: h5py.Dataset) -> str:
    """Decode string data from an h5py dataset, handling various storage formats."""
    raw_data = dataset[()]

    if isinstance(raw_data, bytes):
        return raw_data.decode("utf-8")
    elif isinstance(raw_data, str):
        return raw_data
    elif isinstance(raw_data, np.ndarray):
        if raw_data.dtype == np.uint8:
            return raw_data.tobytes().decode("utf-8").rstrip("\x00")
        elif raw_data.dtype.kind in ('S', 'U', 'O'):
            item = raw_data.item() if raw_data.ndim == 0 else raw_data[0]
            if isinstance(item, bytes):
                return item.decode("utf-8")
            return str(item)
        else:
            return str(raw_data)
    else:
        return str(raw_data)


def _open_h5_with_retry(path, max_retries: int = 3, base_delay: float = 0.5) -> h5py.File:
    """Open HDF5 file with retry + exponential backoff for shared filesystem resilience.

    Handles transient OSError/RuntimeError from NFS/Weka file locking or I/O issues.
    Returns an open h5py.File handle — caller is responsible for closing it.
    """
    for attempt in range(max_retries + 1):
        try:
            return h5py.File(path, "r")
        except (OSError, RuntimeError) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(
                f"HDF5 open failed (attempt {attempt + 1}/{max_retries + 1}): "
                f"{path} - {e}, retrying in {delay:.1f}s"
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Action padding
# ---------------------------------------------------------------------------

def _pad_action_chunk(
    actions: np.ndarray,
    target_start: int,
    target_end: int,
    actual_start: int,
    actual_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad an action chunk to the target range, returning padded array and boolean mask.

    Returns:
        (padded_actions, is_pad) where is_pad[i] is True for padded (invalid) timesteps.
    """
    chunk_size = target_end - target_start
    action_dim = actions.shape[-1] if len(actions) > 0 else 0

    padded = np.zeros((chunk_size, action_dim), dtype=np.float32)
    is_pad = np.ones(chunk_size, dtype=np.bool_)

    if len(actions) > 0:
        start_offset = actual_start - target_start
        end_offset = start_offset + len(actions)
        padded[start_offset:end_offset] = actions
        is_pad[start_offset:end_offset] = False

    return padded, is_pad
