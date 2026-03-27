import logging
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
from olmo.util import flatten_lists

from olmo import tokenizer
from olmo.preprocessing.preprocessor_utils import TensorSpec, VariablePaddingSpec
from olmo.tokenizer import get_special_token_ids

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,  
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def _collate(tensors, max_shape=None, dtype=None, pad=None, pad_value=-1, allow_truncate=True):
    batch_shape = np.stack([x.shape for x in tensors if x is not None], 0).max(0)
    if pad == "to_max":
        row_shape = np.array(max_shape)
        assert np.all(batch_shape[1:] <= row_shape[1:])
        if not allow_truncate:
            if batch_shape[0] > row_shape[0]:
                import pdb; pdb.set_trace()
            assert batch_shape[0] <= row_shape[0]
    elif pad is None:
        row_shape = batch_shape
    else:
        raise NotImplementedError(pad)

    # get the max per dim for all the dims in [1:] in tensor
    tensor = [x for x in tensors if x is not None][0]
    arr = np.full([len(tensors)] + row_shape.tolist(), pad_value,
                  dtype=dtype or tensor.dtype)
    for ix, tensor in enumerate(tensors):
        if tensor is not None:
            t = tensor[:row_shape[0]]
            slices = tuple(slice(None, dim) for dim in t.shape)
            arr[(ix,) + slices] = t
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""
    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]

    def __init__(self, special_tokens,
                 shapes_to_pad_to: Optional[Dict[str, Union[VariablePaddingSpec, TensorSpec]]]=None,
                 include_metadata=True, pad=None, skip_padding=None, cp_enabled=False):
        """
        :param max_text_len: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert shapes_to_pad_to is not None
        self.shapes_to_pad_to = shapes_to_pad_to
        self.include_metadata = include_metadata
        self.pad = pad
        self.cp_enabled = cp_enabled
        self._special_tokens = np.array([
            special_tokens[tokenizer.IM_END_TOKEN],
            special_tokens[tokenizer.IM_START_TOKEN],
            special_tokens[tokenizer.IM_COL_TOKEN],
            special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN],
            special_tokens[tokenizer.IMAGE_PATCH_TOKEN],
        ])[None, :]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        if self.pad is not None:
            max_sequence_len = self.shapes_to_pad_to["tokens"].shape[0]
            # Sanity checks
            for ex in batch:
                if np.any(self._special_tokens == ex["input_tokens"][max_sequence_len:][:, None]):
                    raise ValueError("An image would have gotten truncated!")
                if not self.cp_enabled:
                    ## In CP, as a device might only process image + prompt tokens and no response tokens where the loss 
                    ## would be zero whch is ok.
                    if np.any(ex["loss_masks"] != 0) and np.all(ex["loss_masks"][:max_sequence_len] == 0):
                        raise ValueError("All loss tokens truncated!")
        else:
            max_sequence_len = None

        out = {}
        for key in self.TEXT_KEYS:
            # If one example has subsegment_ids, all examples need it as well
            # Note it is okay if some batches have subsegment_ids and some (for different devices)
            # don't since it only used to modify the attention mask
            if key == "subsegment_ids":
                if any(key in ex for ex in batch):
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])
                else:
                    continue
            dtype = np.float32 if key == "loss_masks" else np.int64
            out[key] = _collate(
                [ex.get(key) for ex in batch], [max_sequence_len], dtype, pad=self.pad)

        for key, spec in self.shapes_to_pad_to.items():
            if key == "tokens":
                continue
            tensors = [ex.get(key) for ex in batch]
            if all(x is None for x in tensors):
                if self.pad is not None:
                    # Create an all-padding input, we might need this to make sure each device
                    # in a FSDP setup gets the same inputs
                    out[key] = torch.full(
                        [len(tensors)] + list(spec.shape), -1,
                        dtype=numpy_to_torch_dtype_dict[spec.dtype],
                    )
            else:
                if isinstance(spec, VariablePaddingSpec):
                    pad = None
                else:
                    pad = self.pad
                pad_value = 0 if spec.dtype == np.uint8 else -1
                out[key] = _collate([ex.get(key) for ex in batch], spec.shape,
                                        dtype=spec.dtype, pad=pad, pad_value=pad_value, allow_truncate=False)

        def _collate_action_chunks() -> Optional[Dict[str, torch.Tensor]]:
            has_chunks = any(
                (ex.get("packed_states") is not None or
                 ex.get("packed_actions") is not None)
                for ex in batch
            )
            if not has_chunks:
                return None

            def _ensure_count(current: Optional[int], arr: np.ndarray, key: str) -> int:
                if current is None:
                    return arr.shape[0]
                if arr.shape[0] != current:
                    raise ValueError(f"Inconsistent chunk counts for '{key}': expected {current}, got {arr.shape[0]}")
                return current

            state_chunks: List[np.ndarray] = []
            action_chunks: List[np.ndarray] = []
            pad_chunks: List[np.ndarray] = []
            batch_ids: List[np.ndarray] = []
            example_ids: List[np.ndarray] = []

            for batch_ix, example in enumerate(batch):
                num_chunks: Optional[int] = None
                state_arr = example.get("packed_states")
                if state_arr is not None:
                    arr = np.asarray(state_arr, dtype=np.float32)
                    num_chunks = _ensure_count(num_chunks, arr, "states")
                    state_chunks.append(arr)
                action_arr = example.get("packed_actions")
                if action_arr is not None:
                    arr = np.asarray(action_arr, dtype=np.float32)
                    num_chunks = _ensure_count(num_chunks, arr, "actions")
                    action_chunks.append(arr)
                pad_arr = example.get("packed_action_is_pad")
                if pad_arr is not None:
                    arr = np.asarray(pad_arr, dtype=np.bool_)
                    num_chunks = _ensure_count(num_chunks, arr, "action_is_pad")
                    pad_chunks.append(arr)
                example_arr = example.get("packed_example_ids")
                if example_arr is not None:
                    arr = np.asarray(example_arr, dtype=np.int64)
                    num_chunks = _ensure_count(num_chunks, arr, "packed_example_ids")
                    example_ids.append(arr)
                elif num_chunks is not None:
                    example_ids.append(np.zeros(num_chunks, dtype=np.int64))
                if num_chunks is None:
                    continue
                batch_ids.append(np.full(num_chunks, batch_ix, dtype=np.int64))

            if not batch_ids:
                return None

            tensors: Dict[str, torch.Tensor] = {}
            if state_chunks:
                tensors["states"] = torch.from_numpy(np.concatenate(state_chunks, axis=0)).to(torch.float32)
            if action_chunks:
                tensors["actions"] = torch.from_numpy(np.concatenate(action_chunks, axis=0)).to(torch.float32)
            if pad_chunks:
                tensors["action_is_pad"] = torch.from_numpy(np.concatenate(pad_chunks, axis=0)).to(torch.bool)
            tensors["packed_batch_idx"] = torch.from_numpy(np.concatenate(batch_ids, axis=0)).to(torch.long)
            tensors["packed_example_ids"] = torch.from_numpy(np.concatenate(example_ids, axis=0)).to(torch.long)
            return tensors

        def _stack_optional_dense(key: str, torch_dtype: torch.dtype, numpy_dtype=None):
            values = [ex.get(key) for ex in batch]
            present = [v is not None for v in values]
            if not any(present):
                return
            if not all(present):
                raise ValueError(f"Examples in batch are missing key '{key}'")
            arrays = []
            for value in values:
                arr = np.asarray(value)
                if numpy_dtype is not None:
                    arr = arr.astype(numpy_dtype, copy=False)
                arrays.append(arr)
            first_shape = arrays[0].shape
            if any(arr.shape != first_shape for arr in arrays[1:]):
                raise ValueError(f"Inconsistent shapes for '{key}': {[arr.shape for arr in arrays]}")
            stacked = np.stack(arrays)
            tensor = torch.from_numpy(stacked)
            out[key] = tensor.to(torch_dtype)

        chunk_tensors = _collate_action_chunks()
        if chunk_tensors is not None:
            out.update(chunk_tensors)
        else:
            _stack_optional_dense("states", torch.float32, numpy_dtype=np.float32)
            _stack_optional_dense("actions", torch.float32, numpy_dtype=np.float32)
            _stack_optional_dense("action_is_pad", torch.bool, numpy_dtype=np.bool_)

        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
