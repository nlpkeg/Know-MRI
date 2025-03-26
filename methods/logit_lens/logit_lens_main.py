from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from . import LogitLensHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd

def get_attributes(x: nn.Module, attributes: str):
    """
    Gets a list of period-separated attributes.
    i.e., get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

lock_kn = threading.Lock()

def diagnose(sample, model_name_or_path, hparams=None):
    """
    Return: dic: {"output": ground_truth of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = ""
    result["image"] = []
    result["table"] = []

    with lock_kn:
        # Method preparation
        hparams = LogitLensHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        output_embed = mt.model.get_output_embeddings()
        if output_embed is not None:
            embedding_weight = output_embed.weight
        else:
            # For models like ChatGLM
            try:
                embedding_weight = get_attributes(mt.model, "transformer.embedding.word_embeddings.weight")
            except AttributeError:
                raise ValueError("Cannot find embedding weight")

        prob_dic_fntoken = []
        prompt = sample["prompt"]

        if prompt not in mt.cache_hiddenstates:
            with torch.no_grad():
                inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)

                # Conditionally handle T5 models
                if "t5" in model_name_or_path.lower():
                    # For T5 models, add decoder_input_ids
                    decoder_input_ids = torch.tensor(
                        [[mt.model.config.decoder_start_token_id]],
                        device=mt.model.device
                    )
                    model_output = mt.model(
                        **inputs,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True
                    )
                    # Use decoder hidden states for T5
                    hidden_states = model_output.decoder_hidden_states
                else:
                    # For non-T5 models, use standard forward pass
                    model_output = mt.model(**inputs, output_hidden_states=True)
                    hidden_states = model_output.hidden_states

                # Cache the hidden states
                mt.cache_hiddenstates[prompt] = torch.stack(hidden_states).cpu()

            # Process hidden states
            for hidden_s in mt.cache_hiddenstates[prompt]:
                hidden_states = hidden_s[0, -1, :].detach().clone()
                with torch.no_grad():
                    if "t5" in model_name_or_path.lower():
                        embedding_weight = embedding_weight.to(torch.float32)  # Convert to float32
                        hidden_states = hidden_states.to(torch.float32)  # Ensure hidden_states 
                    logits = embedding_weight @ hidden_states.to(embedding_weight.device)
                    prob = torch.nn.functional.softmax(logits, dim=-1)
                    _, top_token_id = torch.topk(prob, k=hparams.unembedding_num)
                    prob = prob.cpu().tolist()
                prob_dic_fntoken.append(mt.tokenizer.batch_decode(top_token_id))

        # Prepare layer names and results
        layer_name = [f"Layer_{i}" for i in range(len(prob_dic_fntoken))]
        top_token = prob_dic_fntoken[-1][0]
        result["result_des"] = ""
        result["table"].append({
            "table_name": "Hidden states top token",
            "table_list": [{"Layer name": la, "Top tokens": str(li)} for la, li in zip(layer_name, prob_dic_fntoken)],
            "table_des": "We use the lm head to decode the semantic information in the hidden states layer across the layer.",
            "table_res": f"In the forward propagation of the model: \nThe final token predicted by the model is {top_token}."
        })

        result["origin_data"] = {"top_tokens": prob_dic_fntoken}

    return result