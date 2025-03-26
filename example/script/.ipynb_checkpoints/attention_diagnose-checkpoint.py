from diagnose import diagnose
from dataset_process import knowns, ZsRE, counterfact
from models import llama, gpt2, gptj, bert, t5, internlm, baichuan
from methods import attention_weights

# dataset = knowns.KnownsDataset(loc=knowns.default_loc)
# sample = knowns.get_processed_kvs(dataset[0], kn.requires_input_keys)
# dataset = ZsRE.ZsREDataset(loc=ZsRE.default_loc)
# sample = ZsRE.get_processed_kvs(dataset[0], kn.requires_input_keys)
dataset = counterfact.CounterfactDataset(loc=counterfact.default_loc)
sample = counterfact.get_processed_kvs(dataset[0], attention_weights.requires_input_keys)
result = diagnose.diagnosing(sample=sample, model_name_or_path=llama, method=attention_weights.name)

print(result)