import logging
import os
import warnings
from os.path import join

import datasets
import numpy as np

if "MOLMO_DATA_DIR" in os.environ:
    DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "torch_datasets")
    VIDEO_DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "video_datasets")
    MULTI_IMG_DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "multi_image_datasets")
else:
    DATA_HOME = os.environ.get('DATA_HOME', "")
    VIDEO_DATA_HOME = os.environ.get('VIDEO_DATA_HOME', "")
    MULTI_IMG_DATA_HOME = ""


class Dataset:
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get(item, np.random)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, item, rng):
        # `rng` is used to support deterministic data augmentation for tasks that require it.
        # Used to avoid the hazards of relying on the global rng state for determinism
        raise NotImplementedError()


class DeterministicDataset:
    """Dataset wrapper that supports padding and control the random seed based on the epoch"""

    def __init__(self, dataset: Dataset, preprocessor, seed, n_pad=0, weighting=None, preprocessor_kwargs=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.weighting = weighting
        self.seed = seed
        self.n_pad = n_pad
        self.preprocessor_kwargs = preprocessor_kwargs if preprocessor_kwargs else {}

    def __len__(self):
        return len(self.dataset) + self.n_pad

    def __getitem__(self, idx):
        return self.get(idx, 0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, idx, epoch=0):
        rng = np.random.RandomState(
            (self.seed * 195172 + idx + len(self.dataset)*epoch) % (2 ** 32 - 1))
        if idx >= len(self.dataset):
            # Padding example
            item = self.dataset.get(0, rng)
            item = dict(item, metadata=dict(item.get("metadata", {}), valid=False))
        else:
            item = dict(self.dataset.get(idx, rng))
        if self.weighting:
            item["weight"] = self.weighting
        if self.preprocessor:
            item = self.preprocessor(item, rng, **self.preprocessor_kwargs)
        return item


class DatasetBase(Dataset):
    def __init__(self, split, sample: int=None):
        super().__init__()
        self.split = split
        self.sample = sample
        data = self.load()
        if sample is not None:
            data = data[:self.sample]
        self.data = data

    def load(self):
        raise NotImplementedError()

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def get(self, item, rng):
        raise NotImplementedError()


class HfDataset(Dataset):
    PATH = None

    @classmethod
    def download(cls, n_procs=None):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()

    def __init__(self, split: str, keep_in_memory=True, **kwargs):
        self.split = split
        self.dataset = datasets.load_dataset(
            self.PATH, split=split, keep_in_memory=keep_in_memory, **kwargs)

    def __len__(self):
        return len(self.dataset)
