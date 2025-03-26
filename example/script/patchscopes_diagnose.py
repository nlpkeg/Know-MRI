from diagnose import diagnose
from dataset_process import knowns, ZsRE, counterfact, pararel, PEP3k, TwentyQ
from models import llama, gptj, gpt2, qwen, internlm, baichuan
from methods import patchscopes
import methods
dataset = knowns.KnownsDataset(loc=knowns.default_loc)
sample = knowns.get_processed_kvs(dataset[0], patchscopes.requires_input_keys)
result = diagnose.diagnosing(sample=sample, model_name_or_path=gptj, method=patchscopes.name)

print(result)
