from util.model_tokenizer import get_cached_model_tok
from .SPINE_TRAIN.spine_model import SPINEModel
from . import SPINEHyperParams
import os 
import json
import numpy
import torch
from pathlib import Path
import logging
import os
from .SPINE_TRAIN.spine_train import Solver
temp_dir = Path(__file__).parent.parent.parent/"util"/"tmp"
temp_dir.mkdir(exist_ok=True)
def diagnose(sample, model_name_or_path, hparams=None):
    dtype = torch.float32 if torch.cuda.is_available() else torch.float32
    device = "cuda:0"
    hparams = SPINEHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
    topk = hparams.top_k
    model_name = hparams.model_path
    mt = get_cached_model_tok(model_name=hparams.model_path)
    tokenizer = mt.tokenizer
    model = mt.model
    best_model_path =temp_dir/f"{hparams.model_path.replace('/', '_')}_epoch{hparams.epoch}_lr{hparams.lr}_asl{hparams.asl}_psl{hparams.psl}_hidden_dim{hparams.hidden_dim}_noise{hparams.noise}_mean_value{hparams.mean_value}.pth"

    final_result = {}
    prompt = sample["prompt"]
    words = prompt.split()
    tokens1 = []
    embeddings1 = []
   
    for word in words:
        inputs = tokenizer(word, return_tensors="pt").to(device)#  [1, sequence_length]
        outputs = model.get_input_embeddings()(inputs["input_ids"])# [1, sequence_length, embedding_dim]
        embeddings = outputs[0]# [sequence_length, embedding_dim]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])# [sequence_length]
        tokens1.append(tokens)
        embeddings1.append(embeddings)

    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight.float().cpu().detach().numpy()
    
    solver = Solver(embedding_matrix,hparams=hparams)
    vocab = tokenizer.get_vocab()
    inv_vocab = {index: token for token, index in vocab.items()}
    all_embeddings = torch.tensor(embedding_matrix).to(dtype).to(device)
    if os.path.isfile(best_model_path):
        spine_model = SPINEModel(input_dim=embedding_matrix.shape[1],hparams=hparams)
        spine_model.load_state_dict(torch.load(best_model_path))
        spine_model.to(device)
        spine_model.eval()
        with torch.no_grad():
            for token, embedding in zip(tokens1, embeddings1):
                batch_x = torch.tensor(embedding).to(dtype)
                batch_x = batch_x.clone().detach().to(device)
                batch_y = batch_x.clone().detach().to(device)
                _,h,_,_ = spine_model(batch_x, batch_y)#h = [[1,2,3],[1,2,3]]
                top_values = torch.topk(h, k=topk, dim=-1)[0].tolist()
                top_idxs = torch.topk(h, k=topk, dim=-1)[1].tolist()
                for small_token, values,top_idx in zip(token, top_values,top_idxs):
                    c_lists = []
                    if small_token not in ["[CLS]", "[SEP]"]:
                        for value,idx in zip(values,top_idx):
                            current_list = []
                            current_list.append(idx)
                            current_list.append(value)
                            c_lists.append(current_list)
                            final_result[small_token] =c_lists
            # Get the SPINE embeddings for all tokens in one go
            _, spine_matrix, _, _ = spine_model(all_embeddings, all_embeddings)
            top_values, top_indices = torch.topk(spine_matrix, k=hparams.topk_tokens, dim=-1)
            top_indices = top_indices.cpu().numpy()
            #top_words = [[inv_vocab[idx] for idx in idxs] for idxs in top_indices]
            top_words = [[inv_vocab.get(idx, 'UNK') for idx in idxs] for idxs in top_indices]
           
            
            final_list = []
            tok_topwords = []
            for k,v in final_result.items():
                for item in v:
                    current = [item]
                    tok_topwords.extend(current)
                    tok_topwords.append(top_words[item[0]])
                final_result[k] = tok_topwords
                tok_topwords = []
                                          
    else:
        with torch.set_grad_enabled(True):
            solver.train()
        final_result = solver.return_result(tokens1, embeddings1,topk,all_embeddings,inv_vocab,hparams=hparams)
            #tokens = ["happy", "ness"]
            #embeddings = [[1,2,3],[1,2,3]]
            # final_result = {'happy': [[1.0, 1.0]], 'ness': [[1.0, 1.0]]}
    original_data = {"sample_tokens":[],"topk_embeddings":[]}
    table = []
    for k,v in final_result.items():
        current_table = {"table_name":f" token {k}","table_list":[],
        "table_des":"Each row represents one of the top-k activation values along with its corresponding dimensional index and the several words from the vocabulary that have the highest activation values in this dimension","table_res":""}
        original_data["sample_tokens"].append(k)
        current_tabellist = []
        v_values = []
        for i in range(0,len(v),2):
            v_values.append(v[i])
            current_tabellist.append({"dimension":v[i][0],"activation values":v[i][1],"The tokens with the highest activation values in this dimension of the vocabulary":v[i+1]})
            current_table["table_list"]=current_tabellist
        original_data["topk_embeddings"].append(v_values)
        table.append(current_table)
    
    ffinal_result ={   
    "origin_data": "any", 
    "table": [{"table_name": "xxx1", "table_list": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "tabel_des": "", "tabel_res": ""}, 
                {"table_name": "xxx2", "table_list": [{"a2": 1, "b2": 2}, {"a2": 3, "b2": 4}], "tabel_des": "", "tabel_res": ""}], 
    "result_des": "" 
    }   
    # The results generated using the SPINE sparse encoder:the top-k activation values and their corresponding tokens with the largest activation values in the respective dimensions
    ffinal_result["origin_data"] = original_data
    ffinal_result["table"] = table

    return ffinal_result
            

        
       
    
