"""
Configuration presets for SynthManip datasets and MolmoAct training.

Action data format: Each action key (e.g., joint_pos, joint_pos_rel, ee_delta) contains
a JSON blob with all move groups as sub-keys (e.g., {"arm": [...], "gripper": [...]}).
All move groups should use the same action key since the data is bundled together.
"""
from typing import Dict, List, Optional, Union

ACTION_SPECS: Dict[str, Dict[str, int]] = {
    "RBY1_full": {
        "base": 3,  # x, y, yaw
        "head": 2,  # pan, tilt
        "left_arm": 7,  # 7-DOF arm
        "left_gripper": 2,  # gripper joint actions
        "right_arm": 7,  # 7-DOF arm
        "right_gripper": 2,  # gripper joint actions
        "torso": 6,  # torso joints
    },
    "RBY1_door_opening": {
        "base": 3,  # x, y, yaw (delta)
        "left_arm": 7,  # 7-DOF arm (delta)
        "left_gripper": 1,  # gripper (absolute, squeezed from 2D qpos)
        "right_arm": 7,  # 7-DOF arm (delta)
        "right_gripper": 1,  # gripper (absolute, squeezed from 2D qpos)
    },
    "RBY1_multitask": {
        "base": 3,  # x, y, yaw (delta)
        "left_arm": 7,  # 7-DOF arm (delta)
        "left_gripper": 1,  # gripper (absolute, squeezed from 2D qpos)
        "right_arm": 7,  # 7-DOF arm (delta)
        "right_gripper": 1,  # gripper (absolute, squeezed from 2D qpos)
        "torso": 1,  # torso joint index 1 only (absolute)
    },
    "franka_joint": {
        "arm": 7,
        "gripper": 1,
    },
    "franka_jointdelta": {
        "arm": 7,
        "gripper": 1,
    },
    "franka_eedelta": {
        "ee_delta": 6,  # 3 pos + 3 rot
        "gripper": 1,
    },
    "franka_ee": {
        "ee": 6,  # 3 pos + 3 rot
        "gripper": 1,
    },
}

# Action dataset keys by preset.
# Each key maps to an h5 dataset that contains JSON with all move groups bundled together.
ACTION_DATASET_KEYS: Dict[str, Union[str, Dict[str, str]]] = {
    "RBY1_full": "delta_actions",
    "RBY1_door_opening": {
        "base": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos",
        "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",
    },
    "RBY1_multitask": {
        "base": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos",
        "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",
        "torso": "joint_pos",
    },
    "franka_joint": "joint_pos",
    "franka_jointdelta": "joint_pos_rel",
    "franka_ee_twist": "ee_twist",
    "franka_ee_pose": "ee_pose",
}

# State specs: per-move-group state dimensions. When a preset is absent, falls back to ACTION_SPECS.
# Only needed when state dim differs from action dim for a move group.
STATE_SPECS: Dict[str, Dict[str, int]] = {
    "RBY1_multitask": {
        "base": 3,
        "left_arm": 7,
        "left_gripper": 1,
        "right_arm": 7,
        "right_gripper": 1,
        "torso": 3,  # joints [1, 2, 3] from full 6D torso qpos
    },
}

# State index selection: which raw qpos indices to extract per move group.
# Only needed when a move group uses a subset of the raw qpos vector.
STATE_INDICES: Dict[str, Dict[str, List[int]]] = {
    "RBY1_multitask": {
        "torso": [1, 2, 3],
    },
}

CAMERA_PRESETS: Dict[str, List[str]] = {
    # RBY1
    "RBY1_full_with_head_gopro": ["wrist_camera_r", "head_camera", "wrist_camera_l"],
    "RBY1_right_arm": ["wrist_camera_r", "head_camera"],
    
    # Franka / DROID (original presets)
    "franka_droid_exo_then_wrist": ["exo_camera_1", "wrist_camera"],
    # "franka_droid_wrist_only": ["wrist_camera"],
    # "franka_randomized_wrist_only": ["wrist_camera"],
    # "franka_droid_exo": ["wrist_camera", "exo_camera_1"],
    # "franka_droid_wrist_with_random_exo": ["wrist_camera", "exo_camera_2"],
    # "franka_two_randomized_exo": ["wrist_camera", "exo_camera_1", "exo_camera_2"],
    
    # Franka with new cameras (ZED Mini + randomized exo cameras)
    "franka_one_random_then_wrist": ["randomized_zed2_analogue_1", "wrist_camera_zed_mini"],
    "franka_wrist_only": ["wrist_camera_zed_mini"],
    "franka_droid": ["wrist_camera_zed_mini", "droid_shoulder_light_randomization"],
    "franka_gopro": ["wrist_camera_zed_mini", "randomized_gopro_analogue_1"],
    "franka_one_random": ["wrist_camera_zed_mini", "randomized_zed2_analogue_1"],
    "franka_two_random": ["wrist_camera_zed_mini", "randomized_zed2_analogue_1", "randomized_zed2_analogue_2"],
    "franka_mixed": ["wrist_camera_zed_mini", "droid_shoulder_light_randomization", "randomized_gopro_analogue_1", "randomized_zed2_analogue_1"],
}

def get_action_config(preset_name: str) -> Optional[Dict[str, int]]:
    """Get action spec for a given preset name."""
    return ACTION_SPECS.get(preset_name)

def get_state_config(preset_name: str) -> Optional[Dict[str, int]]:
    """Get state spec for a given preset. Falls back to action spec if not defined."""
    return STATE_SPECS.get(preset_name, ACTION_SPECS.get(preset_name))

def get_state_indices(preset_name: str) -> Dict[str, List[int]]:
    """Get state index selection map for a given preset. Empty dict if not defined."""
    return STATE_INDICES.get(preset_name, {})

def get_action_key(preset_name: str) -> Optional[Union[str, Dict[str, str]]]:
    """Get action key configuration for a given preset name."""
    return ACTION_DATASET_KEYS.get(preset_name)

def get_camera_config(preset_name: str) -> Optional[List[str]]:
    """Get camera names for a given preset name."""
    return CAMERA_PRESETS.get(preset_name)

