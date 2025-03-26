import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent/"data_raw"
default_loc = default_data_raw_dir/"emotion.json"
dataset_info = {
    "name": "Emotion",
    "des":"Emotion dataset contains texts and the emotions expressed in the texts.",
    "dataset_type": ""
}

class emotionDataset(Dataset):
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

support_template_keys = ["prompt", "prompts", "ground_truth"]
def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    for key in keys:
        if key == "prompt":
            kvs[key] = sample["text"]
        elif key == "prompts":
            kvs[key] = [sample["text"]]
        elif key == "ground_truth":
            kvs[key] = sample["emotion"]
    return kvs

def get_default_dataset():
    return emotionDataset(loc=default_loc)


if __name__ == "__main__":
    emotion_set = emotionDataset(default_loc)
    kvs = get_processed_kvs(emotion_set[0], keys=support_template_keys)
    print(kvs)