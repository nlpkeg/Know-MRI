from util.model_tokenizer import get_cached_model_tok
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from . import SEHyperParams
def diagnose(sample, model_name_or_path, hparams=None):
    hparams = SEHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
    model_name = hparams.model_path 
    device = torch.device("cuda:0")
    mt = get_cached_model_tok(model_name=model_name)
    model = mt.model
    # model.to(device)
    tokenizer = mt.tokenizer
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map="auto")
    prompt_str = ' '.join(sample["prompt"]) if isinstance(sample["prompt"], list) else sample["prompt"]
    ground_truth_str = ' '.join(sample["ground_truth"]) if isinstance(sample["ground_truth"], list) else sample["ground_truth"]

#  to str
    prompt_answer = prompt_str + " " + ground_truth_str
    
    prompt1 = f"Assess the correctness of the statement: '{prompt_answer}' Answer True if correct, False otherwise. Then, provide the most critical {hparams.topk_words} words or phrases that influence your assessment."
    result1 = text_generator(prompt1, max_length=hparams.max_length, do_sample=hparams.do_sample, temperature=hparams.temperature, top_k=hparams.top_k, top_p=hparams.top_p, repetition_penalty=hparams.repetition_penalty,num_return_sequences=1)
    result = {   
    "origin_data": [prompt1], 
    "image":"",
    "table": [{"table_name": "Self-explanation of LLM", 
    "table_list": [{"Input prompt": prompt1, "answer": result1[0]['generated_text'].replace(prompt1, "")}],
    "tabel_des": "the most critical top_k words or phrases that influences LLM's assessment", "tabel_res": ""}], 
    "result_des": "" 
    }   
    return result


