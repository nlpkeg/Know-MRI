from diagnose import diagnose
from dataset_process import knowns, ZsRE, counterfact
from models import llama, gptj, gpt2, qwen, bert, t5, chatglm2, internlm, baichuan
from methods import logit_lens

dataset = knowns.KnownsDataset(loc=knowns.default_loc)
sample = knowns.get_processed_kvs(dataset[3], logit_lens.requires_input_keys)
# dataset = ZsRE.ZsREDataset(loc=ZsRE.default_loc)
# sample = ZsRE.get_processed_kvs(dataset[0], kn.requires_input_keys)
# dataset = counterfact.CounterfactDataset(loc=counterfact.default_loc)
# sample = counterfact.get_processed_kvs(dataset[0], logit_lens.requires_input_keys)
result = diagnose.diagnosing(sample=sample, model_name_or_path=gptj, method=logit_lens.name)

print(result)
