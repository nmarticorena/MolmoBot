from typing import Optional, Tuple, Union, Dict

import os
import numpy as np
import math
import cv2

import torch
import unittest
import torch.nn.functional as F

from molmobot_spoc.utils.constants.camera_constants import (
    GOPRO_VERTICAL_FOV,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_CAMERA_WIDTH,
)

DEFAULT_DISTORTION_PARAMETERS = {
    "k1": 0.051,
    "k2": 0.144,
    "k3": 0.015,
    "k4": -0.018,
}

NULL_DISTORTION_PARAMETERS = {
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
}

ALVARO_UNITY_DISTORTION_PARAMETERS = {
    "zoomPercent": 0.49,
    "k1": 0.9,
    "k2": 5.2,
    "k3": -13.0,
    "k4": 16.3,
    "intensityX": 1.0,
    "intensityY": 0.98,
}

ALVARO_UNITY_DISTORTION_PARAMETER_RANGES = {
    "zoomPercent": (0.45, 0.53),
    "k1": (0.8, 1.0),
    "k2": (5.0, 5.4),
    "k3": (-14.0, -12.0),
    "k4": (15.0, 17.0),
    "intensityX": (0.95, 1.05),
    "intensityY": (0.93, 1.03),
}


DEFAULT_CROP_PERCENT = 0.30


# Initialize the global cache variable properly at module level
global _cached_map
_cached_map = None


def get_default_distortion_map() -> np.ndarray:
    """Get the default distortion map for a camera, loading from disk if necessary."""
    global _cached_map
    if _cached_map is None:
        map_path = "utils/constants/default_unity_distortion_map.npy"
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"No default distortion map found at {map_path}. ")
        _cached_map = np.load(map_path)
        # assert that the map is the right size for the current GOPRO_CAMERA_HEIGHT and GOPRO_CAMERA_WIDTH
        assert (
            _cached_map.shape[0] == GOPRO_CAMERA_HEIGHT
            and _cached_map.shape[1] == GOPRO_CAMERA_WIDTH
        ), (
            f"Default distortion map has wrong size: {_cached_map.shape}, expected: {(GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)}"
        )
    return _cached_map


def calc_camera_intrinsics(fov_y, frame_height, frame_width):
    # this functionality is now here to avoid a circularity or duplication issue
    focal_length = 0.5 * frame_height / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length

    c_x = frame_width / 2
    c_y = frame_height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def get_randomized_distortion_parameters(
    distortion_parameters: Optional[dict] = None,
    randomization_factor: float = 0.001,
) -> dict:
    if distortion_parameters is None:
        distortion_parameters = DEFAULT_DISTORTION_PARAMETERS
    randomized_distortion_parameters = {}
    for key, value in distortion_parameters.items():
        randomized_distortion_parameters[key] = value + np.random.uniform(
            -randomization_factor, randomization_factor
        )
    return randomized_distortion_parameters


def get_randomized_unity_distortion_parameters(
    distortion_parameters: Optional[dict] = None,
    randomization_factor: float = 0.5,
) -> dict:
    if distortion_parameters is None:
        distortion_parameters = ALVARO_UNITY_DISTORTION_PARAMETERS
    randomized_distortion_parameters = {}
    # use the ranges from ALVARO_UNITY_DISTORTION_PARAMETER_RANGES and the relative factor to strengthen or weaken the central tendency
    for key, value in distortion_parameters.items():
        range = ALVARO_UNITY_DISTORTION_PARAMETER_RANGES[key]
        # use randomization factor * delta from center to set a uniform range
        randomized_distortion_parameters[key] = value + np.random.uniform(
            -randomization_factor * (value - range[0]),
            randomization_factor * (range[1] - value),
        )
    return randomized_distortion_parameters


def make_distorted_grid(
    H,
    W,
    K,
    distortion_parameters,
    device=None,
    x_normalized=None,
    y_normalized=None,
    r=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if x_normalized is None or y_normalized is None or r is None:
        # Create meshgrid of pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
        )

        # Normalize pixel coordinates using camera intrinsics
        x_normalized = (x - K[0, 2]) / K[0, 0]
        y_normalized = (y - K[1, 2]) / K[1, 1]

        r = torch.sqrt(x_normalized**2 + y_normalized**2)
    else:
        # Ensure the precomputed values are on the correct device
        x_normalized = x_normalized.to(device)
        y_normalized = y_normalized.to(device)
        r = r.to(device)

    # Extract distortion parameters
    k1, k2, k3, k4 = (distortion_parameters[k] for k in ["k1", "k2", "k3", "k4"])

    # Apply distortion
    distortion_factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8
    x_distorted = x_normalized * distortion_factor
    y_distorted = y_normalized * distortion_factor

    # Transform back to pixel coordinates
    x_distorted = x_distorted * K[0, 0] + K[0, 2]
    y_distorted = y_distorted * K[1, 1] + K[1, 2]

    # Normalize coordinates to [-1, 1] for grid_sample
    x_distorted = 2 * (x_distorted / (W - 1)) - 1
    y_distorted = 2 * (y_distorted / (H - 1)) - 1

    # Stack coordinates
    grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    return grid


def warp_image_gpu(
    image,
    K=None,
    distortion_parameters=None,
    crop_percent=DEFAULT_CROP_PERCENT,
    grid=None,
    x_normalized=None,
    y_normalized=None,
    r=None,
    output_shape=None,
):
    B, C, H, W = image.shape
    assert C == 3, "Input image should have 3 channels (RGB)"

    assert H == GOPRO_CAMERA_HEIGHT and W == GOPRO_CAMERA_WIDTH, (
        f"Image should be raw GoPro format, actually {H}x{W}"
    )

    if grid is None:
        assert distortion_parameters is not None, (
            "distortion_parameters must be provided if grid is not"
        )
        assert K is not None, "K must be provided if grid is not"
        grid = make_distorted_grid(
            H,
            W,
            K,
            distortion_parameters,
            device=image.device,
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            r=r,
        )
    grid = grid.repeat(B, 1, 1, 1)  # [B, H, W, 2]
    distorted_image = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    crop_h = int(H * crop_percent)
    crop_w = int(W * crop_percent)
    cropped_image = distorted_image[
        :,
        :,
        crop_h : -crop_h if crop_h > 0 else None,
        crop_w : -crop_w if crop_w > 0 else None,
    ]

    if output_shape is not None:
        cropped_image = F.interpolate(
            cropped_image, size=output_shape, mode="bilinear", align_corners=True
        )

    return cropped_image


def warp_video_gpu(
    video,
    K=None,
    randomize_distortion_parameters=False,
    crop_percent=DEFAULT_CROP_PERCENT,
    output_shape=None,
):
    assert (
        video.shape[2] == GOPRO_CAMERA_WIDTH and video.shape[1] == GOPRO_CAMERA_HEIGHT
    ), "Image should be raw GoPro format"

    if randomize_distortion_parameters:
        distortion_parameters = get_randomized_distortion_parameters()
    else:
        distortion_parameters = DEFAULT_DISTORTION_PARAMETERS

    if K is None:
        K = calc_camera_intrinsics(
            GOPRO_VERTICAL_FOV, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH
        )

    # Check if the video is already a tensor
    if not isinstance(video, torch.Tensor):
        # Convert to tensor and move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_tensor = torch.from_numpy(video).float().to(device) / 255.0
    else:
        video_tensor = video.float() / 255.0

    # Permute to [B, C, H, W] format
    video_tensor = video_tensor.permute(0, 3, 1, 2)

    warped_video = warp_image_gpu(
        image=video_tensor,
        K=K,
        distortion_parameters=distortion_parameters,
        crop_percent=crop_percent,
        output_shape=output_shape,
    )
    # unstack the video
    warped_video = (warped_video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
        np.uint8
    )
    return warped_video


def warp_point(pixel_x, pixel_y, K, distortion_parameters, crop_percent, output_shape):
    # Create a blank frame with the point marked
    blank_frame = torch.zeros(
        (1, 3, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH), dtype=torch.float32
    )
    blank_frame[0, :, int(pixel_y), int(pixel_x)] = 1.0  # Mark the point as white

    # Warp the frame
    warped_frame = warp_image_gpu(
        blank_frame,
        K=K,
        distortion_parameters=distortion_parameters,
        crop_percent=crop_percent,
        output_shape=output_shape,
    )

    # Find the warped point
    warped_frame_np = warped_frame.squeeze().permute(1, 2, 0).cpu().numpy()
    flat_index = np.argmax(warped_frame_np[:, :, 0])
    warped_y, warped_x = np.unravel_index(flat_index, warped_frame_np.shape[:2])

    return warped_x, warped_y


def unwarp_point(pixel_x_distorted, pixel_y_distorted, K, distortion_parameters):
    # Normalize pixel coordinates using camera intrinsics
    x_normalized_distorted = (pixel_x_distorted - K[0, 2]) / K[0, 0]
    y_normalized_distorted = (pixel_y_distorted - K[1, 2]) / K[1, 1]

    # Initial guess for the undistorted coordinates
    x_normalized = x_normalized_distorted
    y_normalized = y_normalized_distorted

    # Extract distortion parameters
    k1, k2, k3, k4 = (distortion_parameters[k] for k in ["k1", "k2", "k3", "k4"])

    # Estimate the undistorted coordinates iteratively
    for _ in range(5):  # Iterate to refine the inverse estimate
        r_distorted = torch.sqrt(x_normalized**2 + y_normalized**2)
        distortion_factor = (
            1
            + k1 * r_distorted**2
            + k2 * r_distorted**4
            + k3 * r_distorted**6
            + k4 * r_distorted**8
        )

        x_normalized = x_normalized_distorted / distortion_factor
        y_normalized = y_normalized_distorted / distortion_factor

    # Transform back to pixel coordinates
    pixel_x_undistorted = x_normalized * K[0, 0] + K[0, 2]
    pixel_y_undistorted = y_normalized * K[1, 1] + K[1, 2]

    return pixel_x_undistorted.item(), pixel_y_undistorted.item()


if __name__ == "__main__":
    unittest.main()
