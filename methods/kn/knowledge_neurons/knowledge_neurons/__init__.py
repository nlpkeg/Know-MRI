from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
try:from transformers import LlamaTokenizer, LlamaForCausalLM
except:pass
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
import torch
from .knowledge_neurons import KnowledgeNeurons
from .data import pararel, pararel_expanded, PARAREL_RELATION_NAMES

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS


def te(model_name: str):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer

def initialize_model_and_tokenizer(model_name: str, dtype="fp32", device_map="auto"):
    torch_dtype = torch.float16 if dtype=="fp16" else torch.float32
    if dtype == "bf16":
        torch_dtype = torch.bfloat16

    if 't5' in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        tok = T5Tokenizer.from_pretrained(model_name)
    elif 'gpt-3.5' in model_name.lower():
        model, tok = None, None
    elif 'gpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        tok = GPT2Tokenizer.from_pretrained(model_name)
        tok.pad_token_id = tok.eos_token_id
    elif 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        tok = LlamaTokenizer.from_pretrained(model_name)
        tok.pad_token_id = tok.eos_token_id
    elif 'baichuan' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.pad_token_id = tok.eos_token_id
    elif 'chatglm' in model_name.lower():
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.unk_token_id = 64787
        # self.tok.pad_token_id = self.tok.eos_token_id
    elif 'internlm' in model_name.lower():
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.pad_token_id = tok.eos_token_id
    elif 'qwen' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, fp32=False, trust_remote_code=True,device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name, eos_token='<|endoftext|>',
                                                        pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                        trust_remote_code=True)
    elif 'mistral' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token_id = tok.eos_token_id
    elif 'bert' in model_name.lower():
        tok = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
        model = model.cuda()
    else:
        raise NotImplementedError
    
    return model, tok


def model_type(model_name: str):
    if "bert" in model_name:
        return "bert"
    elif 'gpt2' in model_name:
        return "gpt2"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    elif 'gpt-j' in model_name or 'gptj' in model_name:
        return 'gptj'
    elif 't5' in model_name:
        return 't5'
    elif 'llama' in model_name:
        return 'llama'
    elif 'baichuan' in model_name.lower():
        return 'baichuan'
    elif 'chatglm2' in model_name.lower():
        return 'chatglm2'
    elif 'internlm' in model_name.lower():
        return 'internlm'
    elif 'qwen2' in model_name.lower():
        return 'qwen2'
    elif 'qwen' in model_name.lower():
        return 'qwen'
    elif 'mistral' in model_name.lower():
        return 'mistral'
    else:
        raise ValueError("Model {model_name} not supported")
