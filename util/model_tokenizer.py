import re
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModel
import torch
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)
try:from transformers import LlamaTokenizer, LlamaForCausalLM
except:pass
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
import torch

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS
def get_attributes(x: torch.nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

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


def initialize_model_and_tokenizer(model_name: str, dtype=torch.float16, device_map="auto"):
    torch_dtype = dtype

    if 't5' in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        tok = T5Tokenizer.from_pretrained(model_name)
    elif 'gpt-3.5' in model_name.lower():
        model, tok = None, None
    elif 'gpt-j' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.pad_token_id = tok.eos_token_id
    elif 'gpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
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
    #elif 'chatglm' in model_name.lower():
        #model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
        #tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        #tok.unk_token_id = 64787
        # self.tok.pad_token_id = self.tok.eos_token_id
    elif 'chatglm' in model_name.lower():
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
        
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not tok.pad_token:
            tok._pad_token = '<|PAD|>' 
            tok.pad_token_id = 0  
        ####
        tok.unk_token_id = 64787
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

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device="auto"
    ):
        print(f"loading model: {model_name}")
        if model is None and tokenizer is None:
            model, tokenizer = initialize_model_and_tokenizer(model_name=model_name, dtype=torch_dtype, device_map=device)
            model.requires_grad_(False)
            model.eval()
        else: 
            raise NotImplementedError
        print(f"{model_name} model loaded")
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.model_type = model_type(model_name=model_name)

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate.dense"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif 'gptj' == self.model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.fc_in"
            self.output_ff_attr = "mlp.fc_out.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif "gpt2" == self.model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'llama' == self.model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'baichuan' == self.model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif "t5" == self.model_type:
            self.transformer_layers_attr = "decoder.block"
            self.input_ff_attr = "layer.2.DenseReluDense.wi_0"
            self.output_ff_attr = "layer.2.DenseReluDense.wo.weight"
            self.word_embeddings_attr = "shared.weight"
        elif 'chatglm2' == self.model_type:
            self.transformer_layers_attr = "transformer.encoder.layers"
            self.input_ff_attr = "mlp.dense_4h_to_h"
            self.output_ff_attr = "mlp.dense_h_to_4h.weight"
            self.word_embeddings_attr = "transformer.embedding.word_embeddings.weight"
        elif 'internlm' == self.model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'qwen2' == self.model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'qwen' == self.model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.w1"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'mistral' == self.model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        else:
            raise NotImplementedError

        self.num_layers = len(get_attributes(self.model, self.transformer_layers_attr))
        self.layer_names = [self.transformer_layers_attr+f".{l}" for l in range(self.num_layers)]

        ix_neuron = 1 if "gpt2" == self.model_type else 0
        self.num_neurons_perlayer = get_attributes(self.model, f"{self.transformer_layers_attr}.0.{self.input_ff_attr}").weight.shape[ix_neuron]
        self.cache_hiddenstates = dict()
        self.cache_attentionweights = dict()

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

import threading
lock_get_cached_model_tok= threading.Lock()
model_name2obj = dict()
def get_cached_model_tok(model_name, model_name2obj=model_name2obj) -> ModelAndTokenizer:
    with lock_get_cached_model_tok:
        if model_name not in model_name2obj:
            model_name2obj[model_name] = ModelAndTokenizer(
                model_name,
                torch_dtype=torch.float16
            )
            if "chatglm" not in model_name:
                model_name2obj[model_name].tokenizer.pad_token = model_name2obj[model_name].tokenizer.eos_token
    return model_name2obj[model_name]