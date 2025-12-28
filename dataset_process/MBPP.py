import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent / "data_raw"
default_loc = default_data_raw_dir / "mbpp_processed.json"
dataset_info = {
    "name": "MBPP",
    "des": """MBPP is a dataset for coding problems.""",
    "dataset_type": ""
}


class MBPPDataset(Dataset):
    def __init__(self, loc: Union[str, Path], *args, **kwargs):
        with open(loc, "r") as f:
            self.data = json.load(f)
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def samples(self, n=5):
        return [get_processed_kvs(self.data[idx]) for idx in range(n)]


support_template_keys = ["prompt", "ground_truth"]


def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    for key in keys:
        if key == "prompt":
            kvs[key] = sample["prompt"]
        elif key == "ground_truth":
            kvs[key] = sample["completion"]

    return kvs


def get_default_dataset():
    return MBPPDataset(loc=default_loc)


if __name__ == "__main__":
    known_set = MBPPDataset(default_loc)
    kvs = get_processed_kvs(known_set[0], keys=support_template_keys)
    print(kvs)