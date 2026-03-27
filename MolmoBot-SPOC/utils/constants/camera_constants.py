import os

ORIGINAL_INTEL_W, ORIGINAL_INTEL_H = 1280, 720
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 396, 224

# This is necessary because the warping lowers the final resolution - final image is 320x240
GOPRO_CAMERA_WIDTH, GOPRO_CAMERA_HEIGHT = 768, 576


# Set Unity warping as the default, use env var to generate flat images if needed
# ex: GENERATE_FLAT_IMAGES=true python your_script.py
GENERATE_FLAT_IMAGES = os.environ.get("GENERATE_FLAT_IMAGES", "").lower() == "true"
USE_UNITY_WARPING = not GENERATE_FLAT_IMAGES

if USE_UNITY_WARPING:
    GOPRO_VERTICAL_FOV = 120  # UNITY SIDE WARPING ONLY
else:
    GOPRO_VERTICAL_FOV = (
        139  # PYTHON SIDE WARPING (generates flat images and torch-warped images)
    )


MODEL_43_WIDTH, MODEL_43_HEIGHT = 320, 240

CAMERAS_TO_WARP = [
    "head_camera",
    "exo_camera_2",
]


def should_warp_camera(sensor_key: str, is_rby1_task: bool = False) -> bool:
    """
    Determine if a camera should be warped.
    """
    if sensor_key in CAMERAS_TO_WARP:
        return True
    elif sensor_key == "first_target_frame_repeated":
        # first_target_frame_repeated should be warped only for rby1 tasks
        # (where it's a copy of head_camera), not for pick and place tasks
        # (where it's a copy of exo_camera_1)
        return is_rby1_task
    return False
