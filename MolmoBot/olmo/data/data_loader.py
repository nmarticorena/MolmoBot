from __future__ import annotations

import logging
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
import omegaconf
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.device_mesh import DeviceMesh

from olmo.config import BaseConfig
from olmo.data.dataset import DeterministicDataset, Dataset
from olmo.data.dynamic_packer import PackingConfig
from olmo.data.get_dataset import get_dataset_by_name
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture
from olmo.io import add_cached_path_clients
from olmo.models.molmo.molmo import MolmoConfig
from olmo.preprocessing.preprocessor_utils import TensorSpec, TOKEN_POOLING_KEYS, \
    VariablePaddingSpec
from olmo.preprocessing.text_preprocessor import MessageWeight
from olmo.torch_util import get_global_rank, get_world_size, get_rank
from olmo.dist_util import get_dp_process_group

log = logging.getLogger(__name__)


@dataclass
class RootSizeMixture(BaseConfig):
    rate: float
    mixture: Dict[str, Optional[float]]


def _init_fn(worker_id):
    add_cached_path_clients()

@dataclass
class KwargsMixture(BaseConfig):
    rate: float
    datasets: List[DatasetWithArgs]
    name: Optional[str] = None  # Name, used only for logging


@dataclass
class DatasetWithArgs(BaseConfig):
    dataset_name: str
    """Data source to use"""

    sampling_rate: Optional[float] = None
    """Flat sampling rate, will be normalized"""

    root_size_factor: Optional[float] = None
    """How to calculate the sampling rate based on the dataset size"""

    message_weight: Optional[MessageWeight] = None
    """Weighting for examples in this dataset"""

    override_p_high_res: Optional[float] = None
    # FIXME maybe it would be better to pass an arg to the preprossor?
    """Override p_high_res in the preprocessor for this dataset"""


@dataclass
class DataLoaderConfig(BaseConfig):
    """Configuration for a torch `DataLoader`"""

    dataset: Optional[str] = None
    """Dataset name, will be used for `get_dataset_by_name`"""

    mixture: Optional[Dict[str, float]] = None
    """Mixture of dataset names and sampling rates"""

    root_size_mixture: Optional[List[RootSizeMixture]] = None
    """Mixture-of-mixtures where sub-mixtures rates are determined by the root dataset size"""

    kwargs_mixture: Optional[List[KwargsMixture]] = None

    split: str = omegaconf.MISSING
    """Dataset split to load"""

    seed: int = omegaconf.MISSING
    """Dataset seed for shuffling and augmentation"""

    pad: Optional[str] = "to_max"
    """How to pad in the collator"""

    sequence_length: Optional[int] = None
    """Max sequence length to truncate examples to in the Collator"""

    max_text_seq_len: Optional[int] = None
    """Max sequence length excluding MM tokens
    
    If set, the sequence_length is computed as `max_text_seq_len` + the max length of the MM tokens
    """

    shuffle: Optional[bool] = True
    """Should the data be shuffled"""

    start_index: int = 0
    """Example index to start at"""

    packing: Optional[PackingConfig] = None

    enable_variable_sized_token_pooling: bool = True

    # DataLoader args
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 300  # 5 minute timeout to catch stuck workers

    def build_eval_dataloader(
        self,
        model_config: MolmoConfig,
        mesh: DeviceMesh,
        batch_size: int,
        for_inference: bool,
        include_metadata: bool = None,
        pad_batches: bool = False,
        max_steps_for_padding=None,
        include_image=False,
    ) -> DataLoader:
        assert self.mixture is None and self.root_size_mixture is None
        log.info(f"Loading eval dataset: {self.dataset}/{self.split}")
        if include_metadata is None:
            include_metadata = for_inference

        dataset = get_dataset_by_name(self.dataset, self.split)
        n_pad = 0
        if mesh is not None:
            dp_process_group = get_dp_process_group(mesh)
            dp_world_size = get_world_size(dp_process_group)
        else:
            dp_process_group = None
            dp_world_size = get_world_size()

        if pad_batches and not self.drop_last:
            global_batch_size = batch_size * dp_world_size
            n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
            if max_steps_for_padding:
                n_steps = min(n_steps, max_steps_for_padding)
            if n_steps*global_batch_size > len(dataset):
                # Pad the dataset so that it can produce enough batches of `global_batch_size` size
                # to cover the entire dataset without dropping any examples
                # We need this if evaluating FSDP models since they will need all devices to get
                # exactly the same number of batches
                n_pad = (n_steps*global_batch_size) - len(dataset)

        if self.pad:
            if not (self.max_text_seq_len or self.sequence_length):
                raise ValueError("Cannot pad without a sequence length set")

        preprocessor = model_config.build_preprocessor(
            for_inference=for_inference,
            is_training=False,
            text_seq_len=self.max_text_seq_len,
            max_seq_len=self.sequence_length,
            include_image=include_image
        )
        dataset = DeterministicDataset(
            dataset=dataset,
            seed=self.seed,
            preprocessor=preprocessor,
            n_pad=n_pad
        )
        sampler = DistributedSampler(
            dataset,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            num_replicas=dp_world_size,
            rank=get_rank(dp_process_group) if dp_process_group is not None else get_global_rank(),
            seed=self.seed,
        )
        output_shapes = preprocessor.get_output_shapes()
        if self.pad:
            output_shape_str = ", ".join(f"{k}={v.shape}" for k, v in output_shapes.items())
            log.info(f"Building eval dataset with output shapes: {output_shape_str}")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=model_config.build_collator(
                output_shapes, self.pad, include_metadata=include_metadata),
            num_workers=self.num_workers,
            worker_init_fn=_init_fn,
            sampler=sampler,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=False if self.num_workers == 0 else self.persistent_workers,
            timeout=self.timeout,
        )

    def build_train_dataloader(
        self,
        model_config: MolmoConfig,
        mesh: DeviceMesh,
        global_batch_size: int,
    ) -> DataLoader:
        """
        Build a training DataLoader with dataset mixing capabilities.

        The function supports three types of dataset mixing, each with different rate calculation methods:
        1. Single dataset: Simple case with rate = 1.0
        2. kwargs_mixture: Complex mixing with root_size_factor and sampling_rate parameters
        3. mixture: Simple mixing with explicit rates
        4. root_size_mixture: Mixing based on square root of dataset sizes

        Rate Calculation Process:
        - Initial rates are computed based on dataset sizes and configuration parameters
        - Rates are normalized within each group (for grouped mixtures)
        - Final rates are normalized across all datasets to sum to 1.0
        - These final rates determine the sampling probability for each dataset during training
        """
        if self.pad:
            if not (self.max_text_seq_len or self.sequence_length):
                raise ValueError("Cannot pad without a sequence length set")
        preprocessor = model_config.build_preprocessor(
            for_inference=False, is_training=True,
            text_seq_len=self.max_text_seq_len, max_seq_len=self.sequence_length)
        if self.dataset:
            ds = get_dataset_by_name(self.dataset, self.split)
            datasets = [DeterministicDataset(ds, preprocessor, self.seed)]
            rates = [1]
        else:
            mixture: Dict[str, Tuple[Dataset, float, Optional[Dict]]] = {}
            if self.kwargs_mixture:
                total_rate = {}
                for group in self.kwargs_mixture:
                    group_datasets = {}
                    for task in group.datasets:
                        t0 = time.perf_counter()
                        log.info(f"Loading train dataset {task.dataset_name}/{self.split}")
                        dataset = get_dataset_by_name(task.dataset_name, self.split)
                        delta = time.perf_counter() - t0
                        if delta > 1:
                            log.info(f"Dataset {task.dataset_name}/{self.split} took {delta:0.1f} seconds")
                        if task.root_size_factor == 0:
                            size = 1
                        elif task.root_size_factor is None:
                            size = np.sqrt(len(dataset))
                        elif task.root_size_factor < 1:
                            size = np.sqrt(len(dataset) * task.root_size_factor)
                        else:
                            size = np.sqrt(task.root_size_factor)
                        if task.sampling_rate is not None:
                            size = size * task.sampling_rate
                        group_datasets[task.dataset_name] = (dataset, size, task)
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*group.rate, w)
                                    for name, (ds, r, w) in group_datasets.items()})
            elif self.mixture:
                for name, rate in self.mixture.items():
                    log.info(f"Loading train dataset {name}/{self.split}")
                    mixture[name] = (get_dataset_by_name(name, self.split), rate, None)
            else:
                for root_size_mixture in self.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        log.info(f"Loading train dataset {name}/{self.split}")
                        dataset = get_dataset_by_name(name, self.split)
                        if as_size is None:
                            size = len(dataset)
                        elif as_size <= 1:
                            size = len(dataset) * as_size
                        else:
                            size = as_size
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate, None)
                                    for name, (ds, r) in group_datasets.items()})

            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            rates = [rate/total_rate for (_, (_, rate, _)) in mixture]
            datasets = []
            for name, (dataset, _, task) in mixture:
                log.info(f"Train dataset {name}/{self.split}: {len(dataset)}")
                if task is not None:
                    task_preprocessor = deepcopy(preprocessor)
                    if task.override_p_high_res:
                        task_preprocessor.preprocessor.image_preprocessor.p_high_res = task.override_p_high_res
                    datasets.append(DeterministicDataset(dataset, task_preprocessor, self.seed, weighting=task.message_weight))
                else:
                    datasets.append(DeterministicDataset(dataset, preprocessor, self.seed))
            log.info("Final sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                log.info(f"{names[ix]}: {100*rates[ix]:0.2f}%")

        output_shapes = preprocessor.get_output_shapes()
        if self.packing:
            # Tell the collator these keys can have a variable shape after padding
            if self.enable_variable_sized_token_pooling:
                for k in TOKEN_POOLING_KEYS:
                    spec = output_shapes.get(k)
                    if spec is not None:
                        output_shapes[k] = VariablePaddingSpec([1]*len(spec.shape), spec.dtype)
                for k in ["num_images", "multimodal_type", "num_image_starts"]:
                    spec = output_shapes.get(k)
                    if spec is not None:
                        output_shapes[k] = VariablePaddingSpec([1], spec.dtype)
            if (model_config.vision_backbone and not
                model_config.vision_backbone.pooling_attention_mask):
                log.warning("Packing should be used with models with pooling_attention_mask=True "
                            "to avoid unexpected test-time behavior")
            if self.packing.max_tokens:
                # lens = dict(input_tokens=self.packing.max_tokens)
                raise NotImplementedError("Packing with manually set max tokens")
            else:
                lens = dict(input_tokens=output_shapes["tokens"].shape[0])

            if "images" in output_shapes:
                if self.packing.max_images:
                    # lens["images"] = self.packing.max_images
                    raise NotImplementedError("Packing with manually set max images")
                else:
                    lens["images"] = output_shapes["images"].shape[0]
            log.info(f"Packing with max-lens: {lens}")
            packer = self.packing.bulid(lens)
        else:
            packer = None

        output_shape_str = ", ".join(f"{k}={v.shape}" for k, v in output_shapes.items())
        log.info(f"Building train dataset with output shapes: {output_shape_str}")
        collator = model_config.build_collator(output_shapes, self.pad, include_metadata=False)
        dataset = IterableDatasetMixture(
            start_index=self.start_index,
            datasets=datasets,
            mixture_rates=rates,
            mesh=mesh,
            global_batch_size=global_batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
            packer=packer
        )
        return DataLoader(
            dataset,
            batch_size=dataset.device_batch_size,
            drop_last=self.drop_last,
            collate_fn=collator,
            worker_init_fn=_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            timeout=self.timeout,
        )

