from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from . import CausalTraceHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import torch
from tqdm import tqdm
import numpy as np
from .causal_tracing import calculate_hidden_flow, collect_embedding_gaussian, collect_embedding_std, collect_embedding_tdist, plot_trace_heatmap


lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    result = dict()
    result["origin_data"] = {"model_output": None, "prob": None, "tokens": None, "subject range": None, 
                             "Restoring state score": None, "Restoring MLP score": None, "Restoring Attn score": None}
    result["image"] = []
    # result["table"] = []
    with lock_kn:
        hparams = CausalTraceHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        uniform_noise = False
        noise_level = hparams.noise_level
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [sample["triple_subject"]]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])
        
        pair_list = []
        for kind in tqdm([None, "attn", "mlp"], desc="Causal tracing..."):
            rt = calculate_hidden_flow(
                            mt,
                            sample["prompt"],
                            sample["triple_subject"],
                            kind=kind,
                            noise=noise_level,
                            uniform_noise=uniform_noise,
                            replace=hparams.replace, # add noise or replace with noice
                            window=hparams.window,
                        )
            numpy_result = {
                    k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v
                    for k, v in rt.items()
                }
            
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
            plot_trace_heatmap(plot_result, savepdf=tmp_png_file, modelname=mt.model_type)
            if not kind:
                name = "Restoring state"
            else:
                kindname = "MLP" if kind == "mlp" else "Attn"
                name = f"Restoring {kindname}"
            img_des = "The above images separately indicate the influence of different hidden layer vectors on the model input." if kind == "mlp" else ""
            score = plot_result["scores"].tolist()
            tem_score = np.array(score)
            max_indices = np.argpartition(tem_score.flatten(), -3)[-3:]
            max_indices = np.unravel_index(max_indices, tem_score.shape)
            max_indices = list(zip(*max_indices))
            tem = [f'Layer_{ind[1]}-Token_{rt["input_tokens"][ind[0]]}' for ind in max_indices]
            img_res = f"For each component, we are computing the top 3 causal tracing scores corresponding to the token-layer pair: \n{', '.join(tem)}."
            result["image"].append({"image_name": name, "image_path": tmp_png_file, "image_des": img_des, "image_res": img_res})
            result["origin_data"]["model_output"] = rt["answer"]
            result["origin_data"]["prob"] = rt["high_score"].item()
            result["output"] = rt["answer"]
            result["origin_data"]["tokens"] = rt["input_tokens"]
            result["origin_data"]["subject range"] = rt["subject_range"]
            result["origin_data"][name+" score"] = score # num tokens * layers
            # pair_list.append([f'Layer_{ind[1]}-Token_{rt["input_tokens"][ind[0]]}' for ind in max_indices])
        # result["result_des"] = f"For each component, we are computing the top 3 causal tracing scores corresponding to the token-layer pair:\nMLP: {pair_list[2]}\nAttention: {pair_list[1]}\nLayer: {pair_list[0]}."
        result["result_des"] = ""
        return result
