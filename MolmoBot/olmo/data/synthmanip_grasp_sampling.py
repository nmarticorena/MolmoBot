"""Grasp-event detection and grasp-aware timestep weighting for SynthManip.

Standalone functions so they can be used by training code, visualization
tools, and unit tests without importing the full dataset class.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import h5py
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GraspEventInfo:
    """Grasp events detected in a single trajectory.

    Attributes:
        traj_length: Number of timesteps in the trajectory.
        gripper_commands: Per-timestep gripper command values.
        grasp_closed_events: Indices where gripper transitions to closed.
        grasp_opened_events: Indices where gripper transitions to open.
        success: Whether the trajectory was successful.
    """
    traj_length: int
    gripper_commands: np.ndarray
    grasp_closed_events: np.ndarray
    grasp_opened_events: np.ndarray
    success: bool


def extract_grasp_events(
    f: h5py.File,
    traj_idx: int,
    traj_length: int,
    gripper_threshold: float = 127.5,
    verbose: bool = False,
) -> GraspEventInfo:
    """Extract grasp open/close events from commanded_action gripper values.

    Gripper semantics: 0 = open, 255 = closed.
    A "close" event fires when the gripper transitions from open → closed (value rises).
    An "open" event fires when the gripper transitions from closed → open (value falls).

    Args:
        f: Open h5 file handle.
        traj_idx: Trajectory index within the file.
        traj_length: Number of timesteps.
        gripper_threshold: Values above this are considered "closed". Default 127.5 for 0-255.
        verbose: If True, emit detailed debug logging.

    Returns:
        GraspEventInfo with detected events.
    """
    traj_key = f"traj_{traj_idx}"

    gripper_commands = np.zeros(traj_length, dtype=np.float32)
    gripper_source = "zeros (default)"

    if verbose:
        log.info(f"[extract_grasp_events] traj_idx={traj_idx}, traj_length={traj_length}, threshold={gripper_threshold}")

    try:
        action_data = f[traj_key]["actions"]
        available_action_keys = list(action_data.keys())

        if verbose:
            log.info(f"[extract_grasp_events] Available action keys: {available_action_keys}")

        if "commanded_action" not in action_data:
            raise KeyError(
                f"'commanded_action' not found in action keys. Available: {available_action_keys}"
            )

        cmd_data = action_data["commanded_action"]

        if verbose:
            log.info(f"[extract_grasp_events] Using 'commanded_action': dtype={cmd_data.dtype}, shape={cmd_data.shape}")

        is_json_bytes = (cmd_data.dtype == np.uint8 and len(cmd_data.shape) == 2)

        if verbose:
            log.info(f"[extract_grasp_events] is_json_bytes={is_json_bytes}")

        sample_values = []
        for i in range(min(traj_length, cmd_data.shape[0])):
            if is_json_bytes:
                byte_array = cmd_data[i]
                json_string = byte_array.tobytes().decode("utf-8").rstrip("\x00")
                frame_data = json.loads(json_string)

                if verbose and i < 3:
                    log.info(f"[extract_grasp_events] Frame {i} keys: {list(frame_data.keys())}")

                if "gripper" in frame_data:
                    gripper_val = frame_data["gripper"]
                    gripper_commands[i] = gripper_val[0] if isinstance(gripper_val, (list, tuple)) else gripper_val
                    gripper_source = "commanded_action['gripper']"
                    if verbose and i < 3:
                        log.info(f"[extract_grasp_events] Frame {i} gripper raw={gripper_val}, stored={gripper_commands[i]}")
                elif verbose and i < 3:
                    log.info(f"[extract_grasp_events] Frame {i} NO 'gripper' key in frame_data")
            else:
                gripper_commands[i] = float(cmd_data[i])
                gripper_source = "commanded_action (native)"

            if i < 5:
                sample_values.append(gripper_commands[i])

        if verbose:
            log.info(f"[extract_grasp_events] First 5 gripper values: {sample_values}")

    except (KeyError, json.JSONDecodeError) as e:
        log.warning(f"Could not extract gripper commands for traj {traj_idx}: {e}")

    if verbose:
        log.info(f"[extract_grasp_events] Gripper source: {gripper_source}")
        log.info(f"[extract_grasp_events] Gripper stats: min={gripper_commands.min():.4f}, max={gripper_commands.max():.4f}, mean={gripper_commands.mean():.4f}")
        unique_vals = np.unique(gripper_commands)
        if len(unique_vals) <= 10:
            log.info(f"[extract_grasp_events] Unique gripper values: {unique_vals}")
        else:
            log.info(f"[extract_grasp_events] {len(unique_vals)} distinct gripper values")

    gripper_closed = gripper_commands > gripper_threshold
    transitions = np.diff(gripper_closed.astype(np.int32))
    grasp_closed_events = np.where(transitions == 1)[0] + 1
    grasp_opened_events = np.where(transitions == -1)[0] + 1

    if verbose:
        log.info(f"[extract_grasp_events] {gripper_closed.sum()} frames closed, "
                 f"{len(grasp_closed_events)} close events at {grasp_closed_events.tolist()}, "
                 f"{len(grasp_opened_events)} open events at {grasp_opened_events.tolist()}")

    success = True
    try:
        if "success" in f[traj_key]:
            success = bool(f[traj_key]["success"][-1])
    except (KeyError, IndexError):
        pass

    if verbose:
        log.info(f"[extract_grasp_events] Trajectory success: {success}")

    return GraspEventInfo(
        traj_length=traj_length,
        gripper_commands=gripper_commands,
        grasp_closed_events=grasp_closed_events,
        grasp_opened_events=grasp_opened_events,
        success=success,
    )


def compute_grasp_aware_weights(
    f: h5py.File,
    traj_idx: int,
    traj_length: int,
    weight_config: Dict[str, Any],
) -> np.ndarray:
    """Compute grasp-aware per-timestep sampling weights for a trajectory.

    Can be used by both SynthmanipDataset and standalone test/visualization code.

    Args:
        f: Open h5 file handle.
        traj_idx: Trajectory index within the file.
        traj_length: Number of timesteps.
        weight_config: Dict with keys:
            lookahead_window (int=2): frames BEFORE event that "see it coming".
            lookback_window  (int=2): frames AFTER event that "look back" at it.
            final_grasp_weight (float=2.0): floor weight for timesteps near the final grasp.
            failed_grasp_weight (float=0.5): multiplicative downweight for failed grasps.
            release_after_failed_grasp_weight (float=3.0): floor weight near release after failure.
            gripper_threshold (float=127.5): threshold for gripper closed detection.
            go_home_weight (float=1.0): floor weight for go-home phase after final release.
            go_home_start_frames (int=5): frames after final release to start go-home window.
            go_home_end_frames (int=20): frames after final release to end go-home window.
            verbose (bool=False): emit detailed debug logging.

    Weight application order:
        Pass 1 — downweights applied multiplicatively  (weights *= failed_grasp_weight)
        Pass 2 — upweights applied as floors           (weights = max(weights, floor))
        Pass 3 — caps enforce upper bound near failed grasps

    Returns:
        Array of shape (traj_length,) with per-timestep weights ≥ 0.
    """
    lookback_window = weight_config.get('lookback_window', 2)
    lookahead_window = weight_config.get('lookahead_window', 2)
    final_grasp_weight = weight_config.get('final_grasp_weight', 2.0)
    failed_grasp_weight = weight_config.get('failed_grasp_weight', 0.5)
    release_after_failed_grasp_weight = weight_config.get('release_after_failed_grasp_weight', 3.0)
    gripper_threshold = weight_config.get('gripper_threshold', 127.5)
    go_home_weight = weight_config.get('go_home_weight', 1.0)
    go_home_start_frames = weight_config.get('go_home_start_frames', 5)
    go_home_end_frames = weight_config.get('go_home_end_frames', 20)
    verbose = weight_config.get('verbose', False)

    weights = np.ones(traj_length, dtype=np.float32)

    if verbose:
        log.info(f"[weight_fn] Computing weights for traj {traj_idx}, length={traj_length}")

    events = extract_grasp_events(f, traj_idx, traj_length, gripper_threshold, verbose=verbose)

    if len(events.grasp_closed_events) == 0:
        if verbose:
            log.info("[weight_fn] No grasp events detected, returning uniform weights")
        return weights

    final_grasp_idx = events.grasp_closed_events[-1]

    if verbose:
        log.info(f"[weight_fn] {len(events.grasp_closed_events)} grasp events, final at idx {final_grasp_idx}")

    failed_grasp_regions = []
    upweight_regions = []

    for i, grasp_idx in enumerate(events.grasp_closed_events):
        is_final = (i == len(events.grasp_closed_events) - 1)
        releases_after = events.grasp_opened_events[events.grasp_opened_events > grasp_idx]
        release_idx = releases_after[0] if len(releases_after) > 0 else None

        start = max(0, grasp_idx - lookahead_window)

        if is_final and events.success:
            end = min(traj_length, grasp_idx + lookback_window + 1)
            upweight_regions.append((start, end, final_grasp_weight))
            if verbose:
                log.info(f"[weight_fn] Grasp {i}: FINAL+SUCCESS at {grasp_idx}, upweight [{start}:{end}] floor={final_grasp_weight}")
        else:
            end = min(traj_length, (release_idx + lookback_window + 1) if release_idx is not None else (grasp_idx + lookback_window + 1))
            failed_grasp_regions.append((start, end))
            if verbose:
                log.info(f"[weight_fn] Grasp {i}: FAILED at {grasp_idx}, downweight [{start}:{end}] *= {failed_grasp_weight}")

            if release_idx is not None:
                rel_start = max(0, release_idx - lookahead_window)
                rel_end = min(traj_length, release_idx + lookback_window + 1)
                upweight_regions.append((rel_start, rel_end, release_after_failed_grasp_weight))
                if verbose:
                    log.info(f"[weight_fn]   -> Release at {release_idx}, upweight [{rel_start}:{rel_end}] floor={release_after_failed_grasp_weight}")

    # Go-home upweighting after final release
    if go_home_weight != 1.0 and events.success and len(events.grasp_closed_events) > 0:
        releases_after_final = events.grasp_opened_events[events.grasp_opened_events > final_grasp_idx]
        if len(releases_after_final) > 0:
            final_release_idx = releases_after_final[0]
            gh_start = final_release_idx + go_home_start_frames
            gh_end = min(traj_length, final_release_idx + go_home_end_frames)
            if gh_start < traj_length:
                upweight_regions.append((gh_start, gh_end, go_home_weight))
                if verbose:
                    log.info(f"[weight_fn] Go-home: final release at {final_release_idx}, upweight [{gh_start}:{gh_end}] floor={go_home_weight}")
        elif verbose:
            log.info(f"[weight_fn] Go-home: no release found after final grasp at {final_grasp_idx}")

    # Pass 1: downweights (multiplicative)
    for start, end in failed_grasp_regions:
        weights[start:end] *= failed_grasp_weight

    # Pass 2: upweights as floors
    for start, end, floor_weight in upweight_regions:
        weights[start:end] = np.maximum(weights[start:end], floor_weight)

    # Pass 3: cap timesteps approaching failed grasps
    for i, grasp_idx in enumerate(events.grasp_closed_events):
        is_final = (i == len(events.grasp_closed_events) - 1)
        if is_final and events.success:
            continue
        approach_start = max(0, grasp_idx - lookahead_window)
        approach_end = grasp_idx + 1
        weights[approach_start:approach_end] = np.minimum(
            weights[approach_start:approach_end], failed_grasp_weight
        )
        if verbose:
            log.info(f"[weight_fn] Failed grasp {i} at {grasp_idx}: capped [{approach_start}:{approach_end}] to max {failed_grasp_weight}")

    if verbose:
        log.info(f"[weight_fn] Final: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}, "
                 f"{np.sum(weights != 1.0)} non-1.0 timesteps")

    return weights
