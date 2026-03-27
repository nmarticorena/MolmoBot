import dataclasses
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from olmo.config import BaseConfig, D
from olmo.preprocessing.preprocessor_utils import TOKEN_POOLING_KEYS
from olmo.preprocessing.multimodal_preprocessor import MultimodalTypes

ACTION_DATA_KEYS = {"states", "actions", "action_is_pad"}
PACKED_ACTION_META_KEYS = {
    "packed_states",
    "packed_actions",
    "packed_action_is_pad",
    "packed_example_ids",
    "packed_num_chunks",
}
ALL_ACTION_KEYS = ACTION_DATA_KEYS | PACKED_ACTION_META_KEYS


def select_subset_2d_knapsack_quantized(t_values, i_values, T, I, t_weight=1.0, i_weight=1.0, t_granularity=1, i_granularity=1):
    """
    Vectorized 2D knapsack with individually quantized constraints to reduce memory usage.
    """
    t_vals = np.array(t_values, dtype=np.int32)
    i_vals = np.array(i_values, dtype=np.int32)
    M = len(t_vals)

    # Quantize constraints and values independently
    T_q = T // t_granularity
    I_q = I // i_granularity

    # Quantize item values (round up to be conservative)
    t_vals_q = (t_vals + t_granularity - 1) // t_granularity
    i_vals_q = (i_vals + i_granularity - 1) // i_granularity

    # Precompute objective values (use original values)
    obj_vals = t_weight * t_vals + i_weight * i_vals

    # DP table with quantized dimensions
    dp = np.zeros((M + 1, T_q + 1, I_q + 1), dtype=np.float32)

    # Vectorized DP fill
    for item in range(1, M + 1):
        t_val_q = t_vals_q[item - 1]
        i_val_q = i_vals_q[item - 1]
        obj_val = obj_vals[item - 1]

        # Copy previous layer
        dp[item] = dp[item - 1]

        # Vectorized update where item can fit
        if t_val_q <= T_q and i_val_q <= I_q:
            # Create shifted view for the "take item" case
            take_val = dp[item - 1, :T_q + 1 - t_val_q, :I_q + 1 - i_val_q] + obj_val

            # Update positions where taking item is better
            dp[item, t_val_q:, i_val_q:] = np.maximum(
                dp[item, t_val_q:, i_val_q:],
                take_val
            )

    # Backtrack to find solution
    selected_indices = []
    t_rem_q, i_rem_q = T_q, I_q

    for item in range(M, 0, -1):
        t_val_q = t_vals_q[item - 1]
        i_val_q = i_vals_q[item - 1]

        if (t_val_q <= t_rem_q and i_val_q <= i_rem_q and
            dp[item, t_rem_q, i_rem_q] != dp[item - 1, t_rem_q, i_rem_q]):
            selected_indices.append(item - 1)
            t_rem_q -= t_val_q
            i_rem_q -= i_val_q
    return selected_indices


EXAMPLE_SUBSEGMENT_INCREMENT = 100000


def _extract_action_chunk(example: Dict[str, Any], example_ix: int) -> Optional[Dict[str, Any]]:
    chunk: Dict[str, Any] = {}
    for key in ACTION_DATA_KEYS:
        if key in example:
            chunk[key] = example.pop(key)
    if not chunk:
        return None
    chunk["example_ix"] = example_ix
    return chunk


def _pack_action_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks or len(chunks) == 0:
        return {}

    def _as_array(value, dtype=None):
        if value is None:
            return None
        arr = np.asarray(value)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    states_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    pad_list: List[np.ndarray] = []
    example_ids: List[int] = []
    for chunk in chunks:
        example_ids.append(int(chunk["example_ix"]))
        states = _as_array(chunk.get("states"), np.float32)
        actions = _as_array(chunk.get("actions"), np.float32)
        pads = chunk.get("action_is_pad")
        if states is not None:
            states_list.append(states)
        if actions is not None:
            actions_list.append(actions)
        if pads is None and actions is not None:
            pads = np.zeros(actions.shape[:-1], dtype=np.bool_)
        if pads is not None:
            pad_list.append(_as_array(pads, np.bool_))

    out: Dict[str, Any] = {}
    if states_list:
        out["packed_states"] = np.stack(states_list, axis=0)
    if actions_list:
        out["packed_actions"] = np.stack(actions_list, axis=0)
    if pad_list:
        out["packed_action_is_pad"] = np.stack(pad_list, axis=0)
    out["packed_example_ids"] = np.asarray(example_ids, dtype=np.int64)
    out["packed_num_chunks"] = np.asarray([len(example_ids)], dtype=np.int32)
    return out


def pack(*examples: Dict) -> Dict:
    keys = set()
    for ex in examples:
        keys.update(ex)
    keys = {k for k in keys if k not in {"metadata"} | ALL_ACTION_KEYS}
    if "subsegment_ids" not in keys:
        keys.add("subsegment_ids")
    patch_keys = [
        k for k in TOKEN_POOLING_KEYS if k in keys]
    if "images" in keys:
        assert len(patch_keys) > 0, "Example had images but no image->token mapping idx"
    image_offset = 0
    action_chunks: List[Dict[str, Any]] = []
    for example_ix, example in enumerate(examples):
        chunk = _extract_action_chunk(example, example_ix)
        if chunk is not None:
            action_chunks.append(chunk)
        # Patch indices need to be offset by total number of images patches
        if "images" in example:
            n_patches = np.prod(example["images"].shape[:2])
            for k in patch_keys:
                if k in example:
                    assert np.all(example[k] < n_patches)
                    example[k] = np.where(example[k] >= 0, example[k] + image_offset, example[k])
            image_offset += n_patches
            assert "position_ids" in example
        n_tokens = len(example["input_tokens"])

        # Modify or add subsegment ids to prevent intra-example attention
        # Tokens can only attend to subsegments >= then their subsegment, so
        # we give example increasing subsegments to prevent cross-example attention
        example_subsegemnt_id = example_ix*EXAMPLE_SUBSEGMENT_INCREMENT
        if "subsegment_ids" not in example:
            example["subsegment_ids"] = np.full([n_tokens], example_subsegemnt_id)
        else:
            example["subsegment_ids"] += example_subsegemnt_id

    if "images" in keys:
        img = next(iter(ex for ex in examples if "images" in ex))["images"]
        for ex in examples:
            if "images" not in ex:
                ex["images"] = np.zeros([0]+list(img.shape[1:]), dtype=img.dtype)

    for ex in examples:
        for key in patch_keys:
            max_pooling_shape = max(ex[key].shape[1] for ex in examples if key in ex)
            if key not in ex:
                ex[key] = np.full([0, max_pooling_shape], -1)
            elif ex[key].shape[1] < max_pooling_shape:
                delta = max_pooling_shape - ex[key].shape[1]
                ex[key] = np.pad(ex[key], [[0, 0], [0, delta]], constant_values=-1)
        for key in ["num_images", "num_image_starts"]:
            if key in patch_keys and key not in ex:
                ex[key] = np.array([0], dtype=np.int64)
        if "multimodal_type" in patch_keys and key not in ex:
            ex["multimodal_type"] = np.array([MultimodalTypes.TEXT_ONLY], dtype=np.int64)
    packed = {k: np.concatenate([ex[k] for ex in examples], axis=0) for k in keys if k != "metadata"}
    packed.update(_pack_action_chunks(action_chunks))
    return packed


def packed_iterator(it, packer):
    for i, ex in enumerate(it):
        out = packer(i, ex)
        if out is not None:
            yield out


@dataclasses.dataclass
class PackingConfig(BaseConfig):
    buffer_size: int = 32
    mode: str = "dynamic_solver"

    text_weight: float = 1.0
    """Text token weight for the dynamic solver"""

    image_weight: float = 1.0
    """Image token weight for the dynamic solver
    
    There are many fewer images than text tokens, but the images also seem easier to optimize since 
    the text usually has variance, so keeping these roughly balanced seems to work based
    """
    shortcut_max_len_images: bool = True

    max_images: Optional[int] = None
    max_tokens: Optional[int] = None

    cp_world_size: Optional[int] = 8

    def bulid(self, max_lens_dict):
        if self.mode == "optimizer_last":
            return OptimizeLast(self.buffer_size, max_lens_dict)
        elif self.mode == "dynamic_solver":
            # FIXME this API
            assert set(max_lens_dict.keys()) == {"input_tokens", "images"}
            if self.shortcut_max_len_images:
                shortcut_if_at_max_len = ["input_tokens", "images"]
            else:
                shortcut_if_at_max_len = ["input_tokens"]
            return DynamicSolver(
                self.buffer_size, max_lens_dict,
                dict(input_tokens=self.text_weight, images=self.image_weight),
                shortcut_if_at_max_len=shortcut_if_at_max_len
            )
        else:
            raise ValueError(self.mode)


class DynamicSolver:
    """Pack examples by running a dynamic program to optimize what examples to pack"""

    def __init__(self, max_buffer_size: int, max_lens: Dict[str, int], weights: Dict[str, float],
                 shortcut_if_at_max_len=(), granularity=512, verbosity=0):
        if len(max_lens) != 2:
            raise NotImplementedError()
        self.max_lens = max_lens
        self.max_buffer_size = max_buffer_size
        self._buffer: List[Dict] = [None for _ in range(max_buffer_size)]
        self._is_empty = list(range(max_buffer_size))
        self.verbosity = verbosity
        self.granularity = granularity
        self.weights = weights
        self.shortcut_if_at_max_len = shortcut_if_at_max_len
        # self._tmp = 0

    def __call__(self, example_id, example) -> List:
        # A bit of hack to use hard-coded names
        k1, k2 = list(self.max_lens)
        m1, m2 = self.max_lens[k1], self.max_lens[k2]
        g1 = (m1 + self.granularity - 1) // self.granularity
        g2 = (m2 + self.granularity - 1) // self.granularity

        pack_lens = {k: len(example.get(k, ())) for k in self.max_lens}
        for k in self.shortcut_if_at_max_len:
            max_len = self.max_lens[k]
            g = (max_len + self.granularity - 1) // self.granularity
            ex_len = pack_lens[k]
            quantized_ex_len = ((ex_len + g - 1) // g)
            if quantized_ex_len >= (max_len // g):
                # This example already hits the constraints, just return it immediately
                # since we won't be able to pack it with anything anyway
                if self.verbosity > 1:
                    example_str = ' '.join(f"{k}={pack_lens[k]}" for k, v in self.max_lens.items())
                    print(f"Example already at constraints: {example_str}")
                return pack(example)

        if self._is_empty:
            # Buffer is not full
            self._buffer[self._is_empty.pop()] = example
            if self.verbosity > 1:
                example_str = ' '.join(f"{k}={pack_lens[k]}" for k, v in self.max_lens.items())
                print(f"Add to pool: {example_str}, empty slots={len(self._is_empty)}")
            return None

        if self.verbosity > 1:
            print(f"Pool1: {[len(x[k1]) for x in self._buffer]}")
            print(f"Pool2: {[len(x[k2]) for x in self._buffer]}")

        # Run the solver on examples in the buffer
        indices = select_subset_2d_knapsack_quantized(
            [len(x.get(k1, ())) for x in self._buffer],
            [len(x.get(k2, ())) for x in self._buffer],
            m1, m2,
            self.weights[k1], self.weights[k2],
            g1, g2
        )
        if len(indices) == 0:
            raise RuntimeError("No indices returned by select_subset_2d_knapsack_quantized")

        # return the selected subset, and retain those indices to be replaced with new examples
        self._is_empty = indices
        if self.verbosity > 0:
            val1 = [len(self._buffer[i].get(k1, ())) for i in indices]
            val2 = [len(self._buffer[i].get(k2, ())) for i in indices]
            print(f"Yield {indices}")
            print(f"{k1}: {sum(val1)} {val1}")
            print(f"{k2}: {sum(val2)} {val2}")

        out = pack(*(self._buffer[i] for i in self._is_empty))
        self._buffer[self._is_empty.pop()] = example
        return out


@dataclasses.dataclass
class PackableExample:
    id: Any
    lens: Dict[str, int]
    size: int
    example: Dict[str, Any]

    def __repr__(self):
        len_str = ",".join(f"{k}={v}" for k,v in self.lens.items())
        return f"{self.id}:{len_str}"


@dataclasses.dataclass
class PackedExamples:
    lens: Dict[str, int]
    size: int
    examples: List[PackableExample]

    @classmethod
    def empty(cls):
        return PackedExamples(defaultdict(int), 0, [])

    def size_with(self, other: PackableExample, max_lens):
        return max((other.lens[k]+self.lens[k])/max_lens[k] for k in max_lens)

    def add(self, other: PackableExample, max_lens):
        self.examples.append(other)
        for k, v in other.lens.items():
            self.lens[k] += v
        self.size = max(self.lens[k]/max_lens[k] for k in max_lens)

    def __repr__(self):
        len_str = ",".join(f"{k}={v}" for k,v in self.lens.items())
        return f"[{len(self.examples)}]:{len_str}"


class OptimizeLast:
    def __init__(self, max_buffer_size: int, max_lens: Dict[str, int], verbosity=0):
        self.max_lens = max_lens
        self.max_buffer_size = max_buffer_size
        self._buffer: List[PackableExample] = []
        self.verbosity = verbosity
        self._in_progression: PackedExamples = PackedExamples.empty()

    def _can_pack_with(self, ex1: Dict, ex2: Dict):
        return all((ex1[k] + ex2[k]) <= max_len for k, max_len in self.max_lens.items())

    def __call__(self, example_id, example) -> List:
        pack_lens = {k: len(example[k]) for k in self.max_lens}
        size = max(pack_lens[k]/self.max_lens[k] for k in self.max_lens)
        packable_example = PackableExample(example_id, pack_lens, size, example)

        if len(self._buffer) < self.max_buffer_size:
            self._buffer.append(packable_example)
            if len(self._buffer) == self.max_buffer_size:
                logging.info("Packing buffer filled")
            if self.verbosity > 1:
                print(f"Adding {packable_example} to buffer")
            return None

        if self.verbosity > 0:
            print(f"Got example: {packable_example}, in-progress: {self._in_progression}")
            if self.verbosity > 1:
                for i, ex in enumerate(self._buffer):
                    print(f"\t{i}: {ex}")

        if self._can_pack_with(self._in_progression.lens, packable_example.lens):
            # Add the new example to in_progress if possible
            self._in_progression.add(packable_example, self.max_lens)
            if self.verbosity > 0:
                print("Adding to in-progress")
            return None

        # We can't just naively pack in the next example, so we will try to pack our in-progress
        # sequence with the largest example we can from the pool
        best_can_pack_with = None
        best_ix = None
        for ix, example in enumerate(self._buffer):
            if (
                (best_can_pack_with is None or best_can_pack_with.size > example.size) and
                self._can_pack_with(self._in_progression.lens, example.lens)
            ):
                best_can_pack_with = example
                best_ix = ix

        if best_ix is not None:
            best_size = self._in_progression.size_with(best_can_pack_with, self.max_lens)
        else:
            best_size = self._in_progression.size

        # If a single example in the buffer is > anything we can produce, yield it
        # This is to stop the buffer doesn't get filled with huge examples we can never
        # pack with anything
        largest_buffer_ix = max(range(len(self._buffer)), key=lambda x: self._buffer[x].size)
        if self._buffer[largest_buffer_ix].size > best_size:
            out = self._buffer[largest_buffer_ix]
            self._buffer[largest_buffer_ix] = packable_example
            if self.verbosity > 0:
                print(f"Yielding buffer example {largest_buffer_ix}: {out} from buffer")
            return pack(out.example)

        if best_ix is None:
            # Nothing we can add to `in_progress`, so we return it
            if self.verbosity > 0:
                print(f"Yielding in-progress {self._in_progression}")
            out = pack(*[x.example for x in self._in_progression.examples])
            self._in_progression = self._in_progression.empty()
            self._in_progression.add(packable_example, self.max_lens)
            return out
        else:
            if self.verbosity > 0:
                print(f"Adding buffered example {best_ix}: {self._buffer[best_ix]} to in-progress")
            self._in_progression.add(self._buffer[best_ix], self.max_lens)
            self._buffer[best_ix] = packable_example
            return None
