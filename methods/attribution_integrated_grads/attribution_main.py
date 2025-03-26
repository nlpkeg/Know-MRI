from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok, ModelAndTokenizer
from . import AttributionHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import torch
from tqdm import tqdm
import numpy as np
from util.nethook import Trace
from .hook import Attribution
import pandas as pd

lock_kn = threading.Lock()

def get_attributes(x: torch.nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


        


def diagnose(sample, model_name_or_path, hparams=None):
    """
    return: dic: {"output": ground_truth of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = sample["ground_truth"]
    result["image"] = []
    with lock_kn:
        # method prepare
        hparams = AttributionHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)

        att = Attribution(mt, model_type=mt.model_type, device="cuda")
        with torch.set_grad_enabled(True):
            score, prompt_list, te = att.get_attribution_scores(prompt=sample["prompt"], ground_truth=sample["ground_truth"], batch_size=hparams.batch_size, steps=hparams.num_steps)
        # len(ground_truth_tokens)*len(prompt_tokens)
        tem_score = np.array(score)
        tem_score = (tem_score - np.min(tem_score)) / (np.max(tem_score) - np.min(tem_score))
        max_indices = np.argpartition(tem_score.flatten(), -3)[-3:]
        max_indices = np.unravel_index(max_indices, tem_score.shape)
        max_indices = list(zip(*max_indices))
        score = tem_score.tolist() 
        prompt_token = mt.tokenizer.batch_decode([[i] for i in prompt_list])   
        ground_truth_token = mt.tokenizer.batch_decode([[i] for i in te])
        # result["result_des"] = f"The attribution scores for the top 3 input-output pairs are: {[f'{prompt_token[ind[1]]}->{ground_truth_token[ind[0]]}' for ind in max_indices]}."

        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure()
        tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
        df = pd.DataFrame(np.array(score), index=ground_truth_token, columns=prompt_token)
        cmap = sns.heatmap(data=df, annot=False, cmap="Reds")
        cmap.set_title(f"Attribution score for output")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel("Prompt tokens")
        plt.ylabel("Ground truth tokens")
        fig = cmap.get_figure()
        fig.savefig(tmp_png_file, transparent=True)
        result["image"].append({"image_name": "Attribution score for output", "image_path": tmp_png_file, 
                                "image_des": "The graph above represents the attribution score of the input prompt's tokens on predicting the tokens in the ground truth.",
                                "image_res": f"The attribution scores for the top 3 input-output pairs are: \n{', '.join([f'{prompt_token[ind[1]]}->{ground_truth_token[ind[0]]}' for ind in max_indices])}."})
        result["origin_data"] = {"Attribution": score, "prompt_tokens": prompt_token, "ground_truth_tokens": ground_truth_token}
       
        return result
