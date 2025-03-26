import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent/"data_raw"
default_loc = default_data_raw_dir/"counterfact.json"

dataset_info = {
        "name": "CounterFact", 
        "des": """CounterFact dataset consists of counterfactual information based on Wikidata.""",
        "dataset_type": ""
    }

class CounterfactDataset(Dataset):
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
    
support_template_keys = ["prompt", "prompts", "ground_truth", "triple_subject", "triple_object"]

def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    for key in keys:
        if key == "prompts":
            kvs[key] = [sample["prompt"]]
        elif key == "ground_truth":
            kvs[key] = sample["target_new"]
        elif key == "triple_subject":
            kvs[key] = sample["subject"]
        elif key == "triple_object":
            kvs[key] = sample["ground_truth"]
    return kvs

def get_default_dataset():
    return CounterfactDataset(loc=default_loc)

if __name__ == "__main__":
    known_set = get_default_dataset()
    kvs = get_processed_kvs(known_set[0], keys=support_template_keys)
    print(kvs)