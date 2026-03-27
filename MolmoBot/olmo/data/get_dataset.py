from olmo.data.dataset import Dataset
from olmo.data.synthmanip_dataset import build_synthmanip_dataset


def get_dataset_by_name(dataset_name, split) -> Dataset:
    if dataset_name.startswith("synthmanip/"):
        return build_synthmanip_dataset(dataset_name, split)

    raise NotImplementedError(dataset_name, split)