from FlagEmbedding import FlagModel
import json
from pathlib import Path
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections, WeightedRanker
    )
from pymilvus import MilvusClient
from dataset_process import dataset_list, name2dataset_module
from tqdm import tqdm

temp_dir = Path(__file__).parent.parent/"util"/"tmp"
temp_dir.mkdir(exist_ok=True)

class Interpret_vectordb:
    def __init__(self, settingname):
        with open(settingname, "r", encoding="utf-8") as f:
            self.setting = json.load(f)
        f.close()
        # self.collection = chromadb.PersistentClient(path=self.setting["vectordb_path"])
        self.bge_model = FlagModel(self.setting["embedding_model_path"], use_fp16=False)
        self.client = MilvusClient(str(temp_dir) + "/" + self.setting["vectordb_path"])
        if not self.client.has_collection(collection_name="Interpret_lm"):
            print("---Creating vectordb---")
            self.client.create_collection(collection_name="Interpret_lm", dimension=self.setting["emb_dim"], metric_type="IP", consistency_level="Strong", auto_id=True)
            self.emb_dataset()
        else:
            print("---Vectordb already exists!---")

    
    def add_data(self, dataset_info, data):
        embeddings = self.bge_model.encode(data["prompt"])
        data = {"vector": embeddings, "data": data, "info": dataset_info["name"]} 
        self.client.insert(collection_name="Interpret_lm", data=data)
        
    def search(self, query):
        embeddings = self.bge_model.encode(query)
        search_res =self.client.search(
            collection_name="Interpret_lm", data=[embeddings], limit=self.setting["topk"], search_params={"metric_type": "IP", "params": {}}, output_fields=["data", "info"]
            )
        return search_res
    def emb_dataset(self):
        for k, module in name2dataset_module.items():
            if k in ["GPT4o_data", "USEREDITINPUT"]:
                continue
            dataset = module.get_default_dataset()
            min_len = min(len(dataset),  self.setting["max_num_pre_set"])
            for data in tqdm(dataset[:min_len], desc=f"Embedding dataset {k}"):
                self.add_data(dataset_info=module.dataset_info, data=module.get_processed_kvs(sample=data, keys=module.support_template_keys))


if __name__ == "__main__":
    data = Interpret_vectordb("/home/liujiaxiang/interpret-lm/embedding_utils/embedding_setting.json")

