from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from . import AttentionWeightsHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
import os 


lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    """
    return: dic: {"output": ground_truth of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = sample["ground_truth"]
    result["image"] = []
    result["table"] = []
    tem_img = []
    with lock_kn:
        # method prepare
        hparams = AttentionWeightsHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        prob_dic_fntoken = dict()
        prompt = sample["prompt"]
        if prompt not in mt.cache_attentionweights:
            with torch.no_grad():
                inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)
                ##qwen
                original_use_cache_quant = None
                if "qwen" in model_name_or_path.lower():
                    if hasattr(mt.model.config, 'use_cache_quantization'):
                        original_use_cache_quant = mt.model.config.use_cache_quantization
                        mt.model.config.use_cache_quantization = True
                ##t5
                if "t5" in model_name_or_path.lower():
                    decoder_start_token_id = mt.model.config.decoder_start_token_id
                    decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(mt.model.device)
                    model_output = mt.model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)
                    attentions = model_output.encoder_attentions
                else:
                    model_output = mt.model(**inputs, output_attentions=True)
                    attentions = model_output.attentions
                ##qwen
                if original_use_cache_quant is not None:
                    mt.model.config.use_cache_quantization = original_use_cache_quant

                mt.cache_attentionweights[prompt] = torch.stack(attentions).cpu()

        attention_weights = [] # num_layer * num_head * num_tokens
        tem_att = []
        for att in mt.cache_attentionweights[sample["prompt"]]:
            # num_layer * batch_size * num_head * num_tokens
            att_perlayer = att[0].tolist()
            attention_weights.append(att_perlayer)
            for h in att_perlayer:
                tem_att.append(-np.var(h))
        orig_tokens = [[t] for t in mt.tokenizer.encode(sample["prompt"])]
        token_list = mt.tokenizer.batch_decode(orig_tokens)

        _, ids = torch.topk(torch.tensor(tem_att), k=hparams.num_heads)
        att_ids = ids.cpu().tolist()
        min_var_id = []

        import seaborn as sns
        import matplotlib.pyplot as plt
        for l, atts in enumerate(attention_weights):
            for h, att in enumerate(atts):
                if (l*len(atts)+h) not in att_ids:
                    continue
                plt.figure()
                tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
                df = pd.DataFrame(att, index=token_list, columns=token_list)
                cmap = sns.heatmap(data=df, annot=False, cmap="Blues")
                cmap.set_title(f"Layer_{l}-Head_{h} attention weights")
                # cmap.set_xlabel("Query tokens", fontsize=20)
                # cmap.set_ylabel("Key tokens", fontsize=20)
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=5)
                fig = cmap.get_figure()
                fig.savefig(tmp_png_file)
                min_var_id.append([l, h])
                tem_img.append({"image_name": f"Layer_{l}-Head_{h} attention weights", "image_path": tmp_png_file})

        result["origin_data"] = {"tokens": token_list, "attention_weight": attention_weights, "min_var_id": min_var_id, "imgs": tem_img}
                  
        return result
