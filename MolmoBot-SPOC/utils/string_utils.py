from typing import Optional

import numpy as np


def convert_byte_to_string(bytes_to_decode: np.ndarray, max_len: Optional[int] = None):
    if max_len is None:
        max_len = bytes_to_decode.shape[-1]
    return (bytes_to_decode.view(f"S{max_len}")[0]).decode()
