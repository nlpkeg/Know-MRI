from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from .knowledge_neurons.knowledge_neurons import FiNEKnowledgeNeurons
from .knowledge_neurons import model_type
from . import FiNEHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import torch
from tqdm import tqdm
import numpy as np
def get_scores(model, tok, sample, hparams):
    fine = FiNEKnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_path),
        device="cuda",
    )
    max_layer = hparams.max_layer if hparams.max_layer != -1 else fine.n_layers()
    scores = fine.get_scores_quick(prompt=sample["prompt"], 
                                            ground_truth=sample["ground_truth"], 
                                            layers=[i for i in range(max_layer)])
    
    _, neuron_flat = torch.topk(scores.cuda().flatten(), k=hparams.num_neuron)
    neuron_set = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in neuron_flat]
    embedding_dim = fine._get_word_embeddings().T # input_dim * vocab_size
    un_emb_dim = model.get_output_embeddings().weight.T
    neuron_dic = {}
    un_neuron_dic = {}
    top_token_list, top_score_list = [], []
    for neuron in tqdm(neuron_set, desc="Getting top tokens for neurons"):
        mlp = fine._get_input_ff_layer(layer_idx=(neuron[0])) # output_dim * input_dim
        neur = mlp[neuron[1], :]
        with torch.no_grad():
            smilarity = neur.to(embedding_dim.device) @ embedding_dim
            try:
                un_sim = neur.to(un_emb_dim.device) @ un_emb_dim
                output = True
            except:
                output = False
        _, index = torch.topk(smilarity, k=hparams.unembedding_num) 
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

    return np.array(scores.cpu()), neuron_dic, neuron_set, un_neuron_dic, [t.replace("<s>", "bos") for t in top_token_list], top_score_list

lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    """
    return: dic: {"output": ground_truth of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = sample["ground_truth"]
    result["image"] = []
    result["table"] = []
    with lock_kn:
        # method prepare
        hparams = FiNEHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        # num_layer * num_neuron list(tu)
        data, neuron_dic, neuron_set, un_neuron_dic, top_token_list, top_score_list = get_scores(model=mt.model, tok=mt.tokenizer, sample=sample, hparams=hparams)
        result["origin_data"] = {"Contribution of the neuron": data.tolist(), "Neuron index": neuron_set, "imgs": []}
        tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
        # tabel = [["Top neurons", "Corresponding top tokens"]]
        table = []
        top_tokens = []
        top_neurons = []
        count = 0
        for k, v in neuron_dic.items():
            if count < 5:
                top_neurons.append(k)
                top_tokens.append(v[0])
            count += 1
            if un_neuron_dic is not None:
                table.append({"Top neurons": k, "Corresponding top tokens": v, "Unemb Corresponding top tokens": un_neuron_dic[k]}) 
            else:
                 table.append({"Top neurons": k, "Corresponding top tokens": v}) 
        # result["result_des"] = f"Through the Fine neuron localization method, we have obtained the following top 5 neuron sets, they are: {list(neuron_dic.keys())[:5]}.\nTheir corresponding semantic information is: {top_token_list[:5]}.\nThe contribution scores are respectively: {top_score_list[:5]}."
        result["result_des"] = ""
        tem = [f'{neu}({tok})' for neu, tok in zip(top_neurons, top_tokens)]
        result["table"].append({"table_name": "Top neuron and relative tokens", "table_list": table[:6], 
                                "table_des": "We decode the semantic information of the neurons ranked in the top 5 using the model's embedding layer and unembedding layer.",
                                "table_res": f"The top neurons (and their meanings) are: {', '.join(tem)}."})
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
        plt.figure()
        cmap = sns.heatmap(data=fn_data, cmap="crest", cbar=False)
        cmap.set_xlabel("Index of neurons", fontsize=10)
        cmap.invert_yaxis()
        cmap.set_ylabel("Layer", fontsize=20)
        fig = cmap.get_figure()
        fig.savefig(tmp_png_file)
        result["origin_data"]["imgs"].append({"image_name": "Contribution of the neuron", "image_path": tmp_png_file})
        return result
