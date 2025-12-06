import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent/"data_raw"
default_loc = default_data_raw_dir/"Zhujiu-World.json"

dataset_info = {
        "name": "Zhujiu-World", 
        "des": """Zhujiu World dataset contains world knowledge questions from Zhujiu Knowledge.""",
        "dataset_type": ""
    }

class ZhujiuWorld(Dataset):
    def __init__(self, loc: Union[str, Path], *args, **kwargs):
        with open(loc, "r") as f:
            self.data = json.load(f)
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def samples(self, n=5):
        return [get_processed_kvs(self.data[idx]) for idx in random.sample(range(len(self)), n)]
    
support_template_keys = ["prompt", "ground_truth"]

def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    choice_data_str = ""
    for label, choice in sample["choices"].items():
        choice_data_str += f"{label}. {choice} "

    kvs["prompt"] = sample["question"] + " " + choice_data_str
    kvs["ground_truth"] = sample["gold_tag"]
    return kvs


def get_default_dataset():
    return ZhujiuWorld(loc=default_loc)

if __name__ == "__main__":
    known_set = get_default_dataset()
    kvs = get_processed_kvs(known_set[0], keys=support_template_keys)
    print(kvs)