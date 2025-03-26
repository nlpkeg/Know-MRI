from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from .knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
from . import KNHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import numpy as np
from tqdm import tqdm
import torch

def get_refined_neurons(model, tok, sample, hparams, mt):
    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_path),
        device="cuda",
    )
    refined_neurons = kn.get_refined_neurons(
        prompts=sample["prompts"],
        ground_truth=sample["ground_truth"],
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )
    scores = kn.score

    neuron_dic = {}
    un_neuron_dic = {}
    top_token_list, top_score_list = [], []
    embedding_dim = kn._get_word_embeddings().T # input_dim * vocab_size
    try:
        un_emb_dim = model.get_output_embeddings().weight.T
    except AttributeError:
    # ChatGLM2
        un_emb_dim = model.transformer.output_layer.weight.T
    _, neuron_flat = torch.topk(scores.cuda().flatten(), k=5)
    neuron_set = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in neuron_flat]
    # output = False
    for neuron in tqdm(neuron_set, desc="Getting top tokens for neurons"):
        if mt.model_type == "chatglm2":
            mlp = kn._get_output_ff_layer(layer_idx=(neuron[0]))
        else:
            mlp = kn._get_input_ff_layer(layer_idx=(neuron[0])) # output_dim * input_dim
        neur = mlp[neuron[1], :]
        with torch.no_grad():
            smilarity = neur.to(embedding_dim.device) @ embedding_dim
            try:
                un_sim = neur.to(un_emb_dim.device) @ un_emb_dim
                output = True
            except:
                output = False
        _, index = torch.topk(smilarity, k=3) 
        token_id = [[i] for i in index.cpu().tolist()]
        token = tok.batch_decode(token_id, skip_special_tokens=False)
        name = f"L{neuron[0]}.U{neuron[1]}"
        neuron_dic[name] = token
        top_score_list.append(scores[neuron].item())
        if output:
            _, un_index = torch.topk(un_sim, k=3)
            un_token_id = [[i] for i in un_index.cpu().tolist()]
            un_token = tok.batch_decode(un_token_id, skip_special_tokens=False)
            un_neuron_dic[name] = un_token
            top_token_list.append(un_token[0])
        else:
            un_neuron_dic = None
    
    return np.array(scores.cpu()), neuron_dic, refined_neurons, un_neuron_dic, [t.replace("<s>", "bos") for t in top_token_list], top_score_list

lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    result = dict()
    result["output"] = sample["ground_truth"]
    result["image"] = []
    result["table"] = []
    with lock_kn:
        # method prepare
        hparams = KNHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        with torch.set_grad_enabled(True):
            data, neuron_dic, neuron_set, un_neuron_dic, top_token_list, top_score_list = get_refined_neurons(model=mt.model, tok=mt.tokenizer, sample=sample, hparams=hparams, mt=mt)
        result["origin_data"] = {"Contribution of the neuron": data.tolist(), "Neuron index": neuron_set, "imgs": []}
        table = []
        top_tokens = []
        top_neurons = []
        for k, v in neuron_dic.items():
            top_neurons.append(k)
            top_tokens.append(v[0])
            if un_neuron_dic is not None:
                table.append({"Top neurons": k, "Corresponding top tokens": v, "Unemb Corresponding top tokens": un_neuron_dic[k]}) 
            else:
                 table.append({"Top neurons": k, "Corresponding top tokens": v}) 
        # result["result_des"] = f"Through the KN neuron localization method, we have obtained the following top 5 neuron sets, they are: {list(neuron_dic.keys())}.\nTheir corresponding semantic information is: {top_token_list}.\nThe contribution scores are respectively: {top_score_list}."
        result["result_des"] = ""
        tem = [f'{neu}({tok})' for neu, tok in zip(top_neurons, top_tokens)]
        result["table"].append({"table_name": "Top neuron and relative tokens", "table_list": table, 
                                "table_des": "We decode the semantic information of the neurons ranked in the top 5 using the model's embedding layer and unembedding layer.",
                                "table_res": f"The top neurons (and their meanings) are: {', '.join(tem)}."})
        tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
        import matplotlib.pyplot as plt
        fn_data = []
        step = 100
        for da in data:
            tem = []
            batch = int(len(da)/step)
            for i in range(batch):
                tem.append(np.max(np.abs(da[i*step: (i+1)*step])))
            fn_data.append(tem)
        import seaborn as sns
        fig = plt.figure()
        cmap = sns.heatmap(data=fn_data, cmap="crest", cbar=False)
        cmap.set_xlabel("Index of neurons", fontsize=10)
        cmap.invert_yaxis()
        cmap.set_ylabel("Layer", fontsize=20)
        fig = cmap.get_figure()
        fig.savefig(tmp_png_file)
        result["origin_data"]["imgs"].append({"image_name": "Contribution of the neuron", "image_path": tmp_png_file})
        return result
