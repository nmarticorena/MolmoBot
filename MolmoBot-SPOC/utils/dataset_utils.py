import torch
import numpy as np


def pad_data(data, start_step: int, end_step: int, data_start: int, data_end: int):
    """
    Args:
        data: The actual loaded data (could be subset of trajectory)
        start_step: Desired start index in trajectory coordinates
        end_step: Desired end index in trajectory coordinates (exclusive)
        data_start: Index of the trajectory where data[0] corresponds to
        data_end: Index of the trajectory where data ends (exclusive, so data[-1] is at data_end-1)
    """
    window_size = end_step - start_step
    is_pad = torch.zeros(window_size, dtype=torch.bool)

    # Mark which positions in the window need padding
    for i, traj_idx in enumerate(range(start_step, end_step)):
        if traj_idx < data_start or traj_idx >= data_end:
            is_pad[i] = True

    # If no padding needed, just return the data
    if not any(is_pad):
        return data, is_pad

    # Calculate how much padding we need
    front_pad_length = max(0, data_start - start_step)
    back_pad_length = max(0, end_step - data_end)

    # Create padding arrays
    pad_shape_front = (front_pad_length,) + data.shape[1:]
    pad_shape_back = (back_pad_length,) + data.shape[1:]

    front_padding = np.zeros(pad_shape_front, dtype=np.float32)
    back_padding = np.zeros(pad_shape_back, dtype=np.float32)

    # Concatenate: [front_padding, data, back_padding]
    padded_data = np.concatenate([front_padding, data, back_padding], axis=0).astype(
        np.float32
    )

    return padded_data, is_pad
