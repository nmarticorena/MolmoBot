"""SynthManip dataset configuration dataclass and global registry.

Separating configuration from the dataset class keeps the dataset file
focused on data-loading logic and makes config objects easy to import
in training scripts without pulling in heavy dataset dependencies.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from olmo.data.robot_processing import RobotProcessorConfig

log = logging.getLogger(__name__)


@dataclass
class SynthmanipDatasetConfig:
    """Configuration for SynthManip dataset.

    Robot-specific fields (camera_names, action_move_group_names, action_spec, action_keys)
    are required and must be explicitly specified — use presets from synthmanip_presets.py
    or provide values directly.
    """

    # ── Required fields (no defaults) ─────────────────────────────────────────
    data_path: str
    """Path to dataset directory.  Must contain ``train/`` and ``val/`` subdirs,
    each with ``house_*/*.h5`` files."""

    camera_names: List[str]
    """Camera names to load frames from.  Use camera presets or specify explicitly."""

    action_move_group_names: List[str]
    """Ordered list of move groups whose values are concatenated into the action vector."""

    action_spec: Dict[str, int]
    """Dimension for each move group."""

    action_keys: Dict[str, str]
    """Mapping from move-group name → h5 key containing that action data."""

    # ── Optional robot-specific overrides ─────────────────────────────────────
    state_spec: Optional[Dict[str, int]] = None
    """Per-move-group state dimensions.  Falls back to action_spec when None.
    Only needed when state dim differs from action dim (e.g. torso: 3 state vs 1 action)."""

    state_indices: Optional[Dict[str, List[int]]] = None
    """Index-selection map for state extraction: move group → raw qpos indices to keep.
    Only needed when a move group uses a subset of the raw qpos vector."""

    # ── Observation / action windowing ────────────────────────────────────────
    input_window_size: int = 1
    """Number of observation frames to use as context."""

    obs_step_delta: int = 8
    """Frame spacing when sampling past frames for ``input_window_size > 1``."""

    action_horizon: int = 16
    """Number of future action steps to predict."""

    use_done_action: bool = False
    """Whether to include the final 'done' action in trajectories."""

    # ── Dataset metadata ──────────────────────────────────────────────────────
    style: str = "demo"
    """Style tag used in prompt formatting."""

    split: str = "train"
    """Dataset split."""

    # ── Normalization ─────────────────────────────────────────────────────────
    robot_processor_config: Optional[RobotProcessorConfig] = None
    """Optional normalization config for actions/states."""

    # ── Weighted sampling ─────────────────────────────────────────────────────
    weighted_sampling: bool = False
    """Use grasp-aware weighted timestep sampling."""

    weight_config: Optional[Dict[str, Any]] = None
    """Configuration for grasp-aware weighting (used when ``weighted_sampling=True``).

    Keys:
        lookahead_window (int=2):   frames BEFORE an event that can "see it coming".
        lookback_window  (int=2):   frames AFTER an event that "look back" at it.
        final_grasp_weight (float=2.0)
        failed_grasp_weight (float=0.5)
        release_after_failed_grasp_weight (float=3.0)
        gripper_threshold (float=127.5)
        go_home_weight (float=1.0)
        go_home_start_frames (int=5)
        go_home_end_frames (int=20)
        verbose (bool=False)

    Weight application order (determines dominance):
        Pass 1 — downweights multiplicative  (weights *= failed_grasp_weight)
        Pass 2 — upweights as floors         (weights = max(weights, upweight))
        Pass 3 — caps near failed grasps     (can override upweights)
    """

    # ── Prompt randomization ──────────────────────────────────────────────────
    randomize_prompts: bool = False
    """Randomize the prompt template for each example."""

    prompt_sampling_randomize_casing: bool = True
    """Randomly lower-case the prompt with 50 % probability."""

    prompt_sampling_randomize_punctuation: bool = True
    """Randomly strip trailing punctuation with 50 % probability."""

    prompt_sampling_prob_threshold: float = 0.1
    """Minimum referral-expression probability to be eligible for sampling."""

    prompt_sampling_temperature: float = 4.0
    """Softmax temperature when sampling from referral expressions (lower = prefer shorter)."""

    # ── Gripper representation ─────────────────────────────────────────────────
    gripper_representation_count: int = 1
    """Number of gripper DOFs exposed to the model (1 or 2)."""

    # ── Object image points ────────────────────────────────────────────────────
    load_object_image_points: bool = False
    """Load object_image_points from ``obs/extra``.

    When enabled, adds ``object_image_points_conditioning``, ``object_image_points_current``,
    and ``conditioning_image`` to the returned example dict.

    HDF5 layout::
        obs/extra/object_image_points/{object}/{camera}/points      # (T, max_points, 2)
        obs/extra/object_image_points/{object}/{camera}/num_points  # (T, 1)
    """

    conditioning_frame: Union[int, str] = 0
    """Which frame to use as the conditioning frame for object_image_points.

    Options:
        ``int``              — use that specific frame index (0 = first frame).
        ``"random_first_10"``— randomly sample from frames 0–9.

    Only used when ``load_object_image_points=True``.
    """

    load_policy_phase: bool = False
    """Load policy_phase from ``obs/extra``.

    Adds ``policy_phase`` (int) and ``policy_phase_name`` (str) to the returned dict.
    """

    # ── Camera selection ───────────────────────────────────────────────────────
    furthest_camera_prob: float = 0.0
    """Probability of selecting the ZED2 camera furthest from the pickup object.

    The two ZED2 analogue cameras are defined by ZED2_CAMERAS in synthmanip_utils.py.
    With probability ``furthest_camera_prob`` the furthest is selected, otherwise the first.
    """

    max_exo_views: int = 1
    """Maximum number of exo (non-wrist) camera views to include."""

    # ── Point prompts ──────────────────────────────────────────────────────────
    use_point_prompts: bool = False
    """Append Molmo html-v2 ``<points>`` tags to the goal string.

    Automatically sets ``load_object_image_points=True``.

    Example output::
        "demo: open the door <points coords=\"1 423 512 2 431 519\">door_handle</points>"
    """

    point_prompt_camera: str = "head_camera"
    """Camera used to extract points for the text prompt (when ``use_point_prompts=True``)."""

    max_points_in_conditioning_frame: int = 1
    """Maximum points returned per object per camera from the conditioning frame."""

    cameras_to_warp: List[str] = field(default_factory=list)
    """Cameras to apply GoPro fisheye warping (resize to 640×480, near-zero barrel distortion).
    Typically ``["head_camera"]`` for RBY1 door opening."""

    # ── Debugging ─────────────────────────────────────────────────────────────
    debug_dir: str = "/tmp/synthmanip_debug"
    """Directory for debug images and logs."""

    def __post_init__(self):
        if self.use_point_prompts and not self.load_object_image_points:
            self.load_object_image_points = True


class SynthmanipConfigRegistry:
    """Global registry for SynthManip dataset configurations.

    Training scripts register a :class:`SynthmanipDatasetConfig` before
    creating data loaders; :func:`build_synthmanip_dataset` retrieves it.

    Usage::

        from olmo.data.synthmanip_config import synthmanip_config_registry, SynthmanipDatasetConfig

        config = SynthmanipDatasetConfig(
            data_path="/path/to/data",
            camera_names=["head_camera"],
            action_move_group_names=["arm", "gripper"],
            action_spec={"arm": 7, "gripper": 1},
            action_keys={"arm": "joint_pos_rel", "gripper": "joint_pos"},
        )
        synthmanip_config_registry.register("synthmanip/my_task", config)
    """

    def __init__(self):
        self._configs: Dict[str, SynthmanipDatasetConfig] = {}

    def register(self, dataset_name: str, config: SynthmanipDatasetConfig) -> None:
        """Register a configuration for *dataset_name*."""
        self._configs[dataset_name] = config
        log.info(f"Registered SynthManip config for '{dataset_name}'")

    def get(self, dataset_name: str) -> Optional[SynthmanipDatasetConfig]:
        """Return the config for *dataset_name*, or None if not found."""
        if dataset_name in self._configs:
            return self._configs[dataset_name]
        # Try stripping optional "synthmanip/" prefix
        if dataset_name.startswith("synthmanip/"):
            base = dataset_name[11:]
            if base in self._configs:
                return self._configs[base]
        return None

    def clear(self) -> None:
        """Remove all registered configurations."""
        self._configs.clear()

    def is_registered(self, dataset_name: str) -> bool:
        """Return True if a config is registered for *dataset_name*."""
        return self.get(dataset_name) is not None


# Module-level singleton — import this in training scripts and the dataset
synthmanip_config_registry = SynthmanipConfigRegistry()


# ---------------------------------------------------------------------------
# Prompt templates
# Maps task type → list of prompt groups. Sample a group uniformly, then
# sample within the group to avoid over-representing similar phrasings.
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATES: Dict[str, List[List[str]]] = {
    "pick_and_place": [
        [
            "Pick up the {pickup_name} and place it in or on the {place_name}.",
            "Pick up the {pickup_name} and place it in the {place_name}.",
            "Pick up the {pickup_name} and place it on the {place_name}.",
        ],
        [
            "Put the {pickup_name} on the {place_name}.",
            "Put the {pickup_name} in the {place_name}.",
            "Put the {pickup_name} in or on the {place_name}.",
        ],
        [
            "Move the {pickup_name} to the {place_name}.",
            "Move the {pickup_name} onto the {place_name}.",
            "Move the {pickup_name} into the {place_name}.",
        ],
        [
            "Grab the {pickup_name} and put it on the {place_name}.",
            "Grab the {pickup_name} and place it in the {place_name}.",
            "Grab the {pickup_name} and drop it on the {place_name}.",
        ],
        [
            "Take the {pickup_name} and set it on the {place_name}.",
            "Take the {pickup_name} to the {place_name}.",
            "Take the {pickup_name} and put it in the {place_name}.",
        ],
        ["Transfer the {pickup_name} to the {place_name}.", "Relocate the {pickup_name} to the {place_name}."],
        [
            "Place the {pickup_name} on the {place_name}.",
            "Place the {pickup_name} in the {place_name}.",
            "Place the {pickup_name} inside the {place_name}.",
        ],
        [
            "Can you move the {pickup_name} to the {place_name}?",
            "Could you put the {pickup_name} on the {place_name}?",
            "Please pick up the {pickup_name} and place it on the {place_name}.",
        ],
        [
            "Bring the {pickup_name} to the {place_name}.",
            "Carry the {pickup_name} over to the {place_name}.",
            "Bring the {pickup_name} over to the {place_name}.",
        ],
        [
            "Get the {pickup_name} and put it on the {place_name}.",
            "Fetch the {pickup_name} and place it in the {place_name}.",
            "Get the {pickup_name} and set it on the {place_name}.",
        ],
        ["{pickup_name} to the {place_name}.", "{pickup_name} on the {place_name}.", "{pickup_name} goes on the {place_name}."],
        [
            "Set the {pickup_name} on the {place_name}.",
            "Set the {pickup_name} down on the {place_name}.",
            "Deposit the {pickup_name} in the {place_name}.",
        ],
        [
            "Drop the {pickup_name} on the {place_name}.",
            "Drop the {pickup_name} in the {place_name}.",
            "Drop the {pickup_name} into the {place_name}.",
        ],
    ],
    "pick": [[
        "Pick up the {pickup_obj_name}.", "Lift the {pickup_obj_name}.",
        "Pick the {pickup_obj_name}.", "Grab the {pickup_obj_name}.", "Get the {pickup_obj_name}.",
    ]],
    "pick_and_place_next_to": [
        [
            "Pick up the {pickup_name} and place it next to the {place_name}.",
            "Pick up the {pickup_name} and place it near the {place_name}.",
        ],
        ["Put the {pickup_name} next to the {place_name}.", "Put the {pickup_name} near the {place_name}."],
        ["Move the {pickup_name} next to the {place_name}.", "Move the {pickup_name} near the {place_name}."],
        [
            "Grab the {pickup_name} and put it next to the {place_name}.",
            "Grab the {pickup_name} and put it near the {place_name}.",
            "Grab the {pickup_name} and place it next to the {place_name}.",
            "Grab the {pickup_name} and place it near the {place_name}.",
            "Grab the {pickup_name} and drop it next to the {place_name}.",
            "Grab the {pickup_name} and drop it near the {place_name}.",
        ],
        [
            "Take the {pickup_name} and set it next to the {place_name}.",
            "Take the {pickup_name} and set it near the {place_name}.",
            "Take the {pickup_name} and put it next to the {place_name}.",
            "Take the {pickup_name} and put it near the {place_name}.",
        ],
        [
            "Transfer the {pickup_name} to be next to the {place_name}.",
            "Transfer the {pickup_name} to be near the {place_name}.",
            "Relocate the {pickup_name} to be next to the {place_name}.",
            "Relocate the {pickup_name} to be near the {place_name}.",
        ],
        ["Place the {pickup_name} next to the {place_name}.", "Place the {pickup_name} near the {place_name}."],
        [
            "Can you move the {pickup_name} to be next to the {place_name}?",
            "Can you move the {pickup_name} to be near the {place_name}?",
            "Could you put the {pickup_name} next to the {place_name}?",
            "Could you put the {pickup_name} near the {place_name}?",
            "Please pick up the {pickup_name} and place it next to the {place_name}.",
            "Please pick up the {pickup_name} and place it near the {place_name}.",
        ],
        ["Get the {pickup_name} and put it next to the {place_name}.", "Fetch the {pickup_name} and place it near the {place_name}."],
        ["Drop the {pickup_name} next to the {place_name}.", "Drop the {pickup_name} near the {place_name}."],
    ],
    "door_open": [
        ["Open the door.", "Pull the door open.", "Push the door open."],
        ["Open the door handle and pull.", "Open the door handle and push.", "Grab the door handle and open the door."],
        ["Can you open the door?", "Please open the door."],
    ],
    "open": [
        ["Open the {pickup_obj_name}.", "Pull the {pickup_obj_name} open.", "Pull open the {pickup_obj_name}."],
        ["Grab the {pickup_obj_name} and open it.", "Grab the {pickup_obj_name} handle and pull."],
        ["Can you open the {pickup_obj_name}?", "Please open the {pickup_obj_name}."],
    ],
}
DEFAULT_PROMPT_TEMPLATES["pick_and_place_color"] = deepcopy(DEFAULT_PROMPT_TEMPLATES["pick_and_place"])
