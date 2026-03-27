from typing import Literal


def camera_name_translation(
    which_camera: Literal["nav", "manip", "front", "left", "right", "down"],
) -> Literal[
    "raw_navigation_camera",
    "raw_manipulation_camera",
    "raw_front_camera",
    "raw_left_camera",
    "raw_right_camera",
    "raw_down_camera",
]:
    if which_camera == "nav":
        return "raw_navigation_camera"
    elif which_camera == "manip":
        return "raw_manipulation_camera"
    elif which_camera == "front":
        return "raw_front_camera"
    elif which_camera == "left":
        return "raw_left_camera"
    elif which_camera == "right":
        return "raw_right_camera"
    elif which_camera == "down":
        return "raw_down_camera"
    else:
        raise ValueError(f"Invalid camera type: {which_camera}")


LIST_OF_CAMERAS = {
    "front": 0,
    "left": 1,
    "right": 2,
    "down": 3,
}
INDEX_TO_CAMERA = {v: k for k, v in LIST_OF_CAMERAS.items()}

TRANSLATED_CAMERA_NAMES = {
    idx: f"{camera_name_translation(name)}_features"
    for idx, name in INDEX_TO_CAMERA.items()
}


class AbstractSensor:
    sensor_uuid = "sensor_name"


class ObservationSensor(AbstractSensor):
    pass


class ManipulationCamera(ObservationSensor):
    sensor_uuid = "raw_manipulation_camera"


class NavigationCamera(ObservationSensor):
    sensor_uuid = "raw_navigation_camera"


class NavigationCameraDuplicate(ObservationSensor):
    sensor_uuid = "raw_navigation_camera_2"


class ManipulationCameraDuplicate(ObservationSensor):
    sensor_uuid = "raw_manipulation_camera_2"


class FrontCamera(ObservationSensor):
    sensor_uuid = "raw_front_camera"


class LeftCamera(ObservationSensor):
    sensor_uuid = "raw_left_camera"


class RightCamera(ObservationSensor):
    sensor_uuid = "raw_right_camera"


class DownCamera(ObservationSensor):
    sensor_uuid = "raw_down_camera"


VISUAL_SENSORS = [
    ManipulationCamera.sensor_uuid,
    NavigationCamera.sensor_uuid,
    NavigationCameraDuplicate.sensor_uuid,
    ManipulationCameraDuplicate.sensor_uuid,
    FrontCamera.sensor_uuid,
    LeftCamera.sensor_uuid,
    RightCamera.sensor_uuid,
    DownCamera.sensor_uuid,
    "first_nav_frame_repeated",
    "first_target_frame_repeated",
    "repeat_goal_viz_on_frame",
    "head_camera",
    "wrist_camera_r",
    "wrist_camera_l",
    "exo_camera_1",
    "wrist_camera",
    "droid_shoulder_light_randomization",
    "wrist_camera_zed_mini",
]


def is_a_visual_sensor(sensor):
    return sensor in VISUAL_SENSORS


def is_a_visual_features(feature):
    og_sensor_name = feature.rsplit("_features", 1)[0]
    return is_a_visual_sensor(og_sensor_name)


def is_a_goal_sensor(sensor):
    goal_sensors = [
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
        "goal_in_camera_2d_first_step",
        "repeat_goal_in_camera_2d",
        "goals",
        "nav_goal_point",
        # "goals_features",
    ]
    return sensor in goal_sensors


def is_a_goal_features(feature):
    goal_sensors = [
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
        "goal_text_features",
        "goal_in_camera_2d_first_step",
        "repeat_goal_in_camera_2d",
        "nav_goal_point",
        "pickup_obj_image_points",
    ]
    return feature in goal_sensors


def is_a_non_visual_sensor(sensor):
    return sensor in [
        "relative_arm_location_metadata",
        "an_object_is_in_hand",
        "last_actions",
        "rooms_seen",
        "room_current_seen",
        "rooms_seen_output",
        "room_current_seen_output",
        "nav_task_relevant_object_bbox",
        "manip_task_relevant_object_bbox",
        "nav_accurate_object_bbox",
        "manip_accurate_object_bbox",
        "goal_in_camera_2d_first_step",
        "repeat_goal_in_camera_2d",
        "nav_goal_point",
    ]


def get_sensor_sizes():
    return {
        # "camera_fovs": 4,
        "camera_rotations": 8,
        # "width": 1,
        # "height": 1,
        "camera_positions": 12,
        "agent_body_params": 3,  # 2 for offset, 3 for collider scale ratio
    }
