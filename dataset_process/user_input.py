import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent/"data_raw"
default_loc = ""
dataset_info = {
    "name": "USEREDITINPUT",
    "des":"Data provided by user",
    "dataset_type": ""
}

class UserInputDataset(Dataset):
    def __init__(self, sample: dict={"prompt": "The capital of China is", "ground_truth": "Beijing"}, loc: Union[str, Path]=None, *args, **kwargs):
        self.data = [sample]
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def samples(self, n=1):
        return [get_processed_kvs(self.data[idx]) for idx in random.sample(range(len(self)), n)]

support_template_keys = ["prompt", "prompts"]
def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    for key in keys:
        if key == "prompts":
            kvs[key] = [sample["prompt"]]
        else:
            kvs[key] = sample[key]
    return kvs

def get_default_dataset():
    return UserInputDataset()


if __name__ == "__main__":
    data_set = UserInputDataset()
    kvs = get_processed_kvs(data_set[0], keys=support_template_keys)
    print(kvs)