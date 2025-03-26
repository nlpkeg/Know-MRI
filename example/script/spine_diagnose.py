from diagnose import diagnose
from dataset_process import ZsRE
from models import llama, gptj, gpt2, bert, t5, qwen, chatglm2, internlm
from methods import SPINE

dataset = ZsRE.ZsREDataset(loc=ZsRE.default_loc)
sample = ZsRE.get_processed_kvs(dataset[0], SPINE.requires_input_keys)
result = diagnose.diagnosing(sample=sample, model_name_or_path=llama, method=SPINE.name)

print(result)