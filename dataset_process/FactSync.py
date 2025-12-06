import json
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
import random

default_data_raw_dir = Path(__file__).parent/"data_raw"
default_loc = default_data_raw_dir/"FactSync.json"

dataset_info = {
        "name": "面向大模型的常识类动态知识探测与编辑数据", 
        "des": """本数据集用于大模型常识类动态知识的探测与编辑，共包含15600 个样本。每个样本围绕 “常识逻辑关系验证与知识关联” 设计，核心数据项涵盖 7 类 prompt（提示文本）及 1 个样本标识，可支撑大模型对常识逻辑的理解、判断与知识编辑能力的评估和训练。""",
        "dataset_type": ""
    }

class FactSyncDataset(Dataset):
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
    
support_template_keys = ["prompt", "prompts", "ground_truth", "triple_subject"]

def get_processed_kvs(sample, keys=support_template_keys):
    kvs = dict(sample)
    kvs["dataset_name"] = dataset_info["name"]
    kvs["dataset_type"] = dataset_info.get("dataset_type", "")
    subject = sample["requested_rewrite"]["subject"]
    ground_truth = sample["requested_rewrite"]["target_new"]["str"]
    kvs["prompt"] = sample["requested_rewrite"]["prompt"].replace("{}", subject).replace(ground_truth, "")
    kvs["prompts"] = [prom.replace(ground_truth, "") for prom in sample["paraphrase_prompts"]]
    kvs["ground_truth"] = ground_truth
    kvs["triple_subject"] = subject
    return kvs

def get_default_dataset():
    return FactSyncDataset(loc=default_loc)

if __name__ == "__main__":
    known_set = get_default_dataset()
    kvs = get_processed_kvs(known_set[0], keys=support_template_keys)
    print(kvs)