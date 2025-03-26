import torch
from util.model_tokenizer import get_cached_model_tok
from . import EPHyperParams
import os 
import json
from pathlib import Path
from util.fileutil import get_temp_file_with_prefix_suffix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import LocallyLinearEmbedding
from accelerate import Accelerator
def isomap(data, num_neighbors=5, num_components=3):
    embedding = Isomap(n_neighbors=num_neighbors, n_components=num_components)
    Y = embedding.fit_transform(data)
    return Y 
def lle(data, num_neighbors=10, num_components=2):
    lle = LocallyLinearEmbedding(n_neighbors=num_neighbors, n_components=num_components, method='standard')
    Y = lle.fit_transform(data)
    return Y
def pca(data, num_components=2):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)  
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(data_standardized)
    
    return reduced_data, scaler

def normalize_tensor(x, dim=1):
    x_norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    return x / x_norm
def diagnose(sample, model_name_or_path, hparams=None):
    dtype = torch.float32 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hparams = EPHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
    top_k = hparams.top_k
    dim = hparams.dim 
    model_name = hparams.model_path
    mt = get_cached_model_tok(model_name=hparams.model_path)
    tokenizer = mt.tokenizer
    model = mt.model
    accelerator = Accelerator()
    input_text = sample["prompt"]
    inputs = tokenizer(input_text, return_tensors="pt")
    model = accelerator.prepare(
    model
)
    model = accelerator.prepare(model)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    input_embeddings = model.get_input_embeddings()(inputs["input_ids"])
  
    vocab_embeddings = model.get_input_embeddings().weight
    
    vocab_embeddings = vocab_embeddings.float()
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-3)

    all_embeddings = []
    original_tokens_positions = []

    for i in range(input_embeddings.shape[1]):
        token_embedding = input_embeddings[0, i, :].unsqueeze(0)
        token_embedding = token_embedding.float()
        similarity_scores = cos(normalize_tensor(token_embedding),normalize_tensor(vocab_embeddings))
        topk_scores, topk_ids = torch.topk(similarity_scores, min(hparams.top_k+1, len(similarity_scores)))
        selected_embeddings = vocab_embeddings[topk_ids[1:]].cpu().detach().numpy()  
        topk_ids = topk_ids[1:] 
        all_embeddings.append(selected_embeddings)  
        original_token = tokenizer.decode([inputs['input_ids'][0][i]])
        original_tokens_positions.append((original_token, token_embedding.cpu().detach().numpy(), topk_ids.cpu().numpy()))
    
    all_embeddings = np.vstack([emb for sublist in all_embeddings for emb in sublist])
    original_embeddings = np.vstack([pos[1] for pos in original_tokens_positions])
    combined_embeddings = np.vstack([original_embeddings, all_embeddings])

    methods = {
        'pca': lambda x, y: pca(x, num_components=y)[0],
        'lle': lambda x, y: lle(x, num_components=y,num_neighbors=hparams.num_neighbors),
        'isomap': lambda x, y: isomap(x, num_components=y,num_neighbors=hparams.num_neighbors)
    }

    method_func = methods.get(hparams.method, lambda x, y: pca(x, num_components=y)[0])
    reduced_embeddings = method_func(combined_embeddings, dim)
    
    original_tokens_reduced = reduced_embeddings[:len(original_tokens_positions)]
    similar_tokens_reduced = reduced_embeddings[len(original_tokens_positions):]
    original_token_coordinates = {}
    for i, (orig_tok, _, _) in enumerate(original_tokens_positions):
        original_token_coordinates[orig_tok] = original_tokens_reduced[i].tolist()
    similar_token_coordinates = {}
    for i, (_, _, topk_ids) in enumerate(original_tokens_positions):
        for j in range(hparams.top_k):  
            idx = i * hparams.top_k + j
            similar_tok = tokenizer.convert_ids_to_tokens([topk_ids[j]])[0]
            similar_token_coordinates[similar_tok] = similar_tokens_reduced[idx].tolist()
    anydata = []
    anydata.append(original_token_coordinates)
    anydata.append(similar_token_coordinates)
    if dim == 2:
        plt.figure(figsize=(14, 12))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        all_tokens_info = []  
       
        for i, (orig_tok, _, topk_ids) in enumerate(original_tokens_positions):
            if i >= hparams.token_nums:
                break
            token_group_info = {"Original Token": orig_tok, "Similar Tokens": []}
           
            if hparams.show_original_points == 1:
                plt.scatter(original_tokens_reduced[i, 0], original_tokens_reduced[i, 1], color='blue', s=50)
            if hparams.show_original_tokens == 1:
                plt.annotate(orig_tok, (original_tokens_reduced[i, 0], original_tokens_reduced[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center',color="blue")
            for j in range(len(topk_ids)):
                idx = i * top_k + j
                similar_tok = tokenizer.convert_ids_to_tokens([topk_ids[j]])[0]
                token_group_info["Similar Tokens"].append(similar_tok)
                if hparams.show_connection_lines == 1:
                    plt.plot([original_tokens_reduced[i, 0], similar_tokens_reduced[idx, 0]],
                [original_tokens_reduced[i, 1], similar_tokens_reduced[idx, 1]], 'r--', alpha=0.3)
                if hparams.show_similar_points == 1:
                    plt.scatter(similar_tokens_reduced[idx, 0], similar_tokens_reduced[idx, 1], color='red', alpha=0.7)
                if hparams.show_similar_tokens == 1:
                    plt.annotate(tokenizer.convert_ids_to_tokens([topk_ids[j]])[0], (similar_tokens_reduced[idx, 0], 
                similar_tokens_reduced[idx, 1]), textcoords="offset points", xytext=(0, 10), ha='center',color="red")
            all_tokens_info.append(token_group_info)
        plt.title(f"Visualization of Tokens and Their Top-K Similar Tokens with {hparams.method.upper()} to {dim}D")
        cell_text = []
        headers = ["Original Tokens", "Similar Tokens"]
        for group in all_tokens_info:
            cell_text.append([group["Original Token"], ", ".join(group["Similar Tokens"])])
       
    
        tmp_png_file = get_temp_file_with_prefix_suffix(prefix=f"method_{hparams.method}_token_nums{hparams.token_nums}_top_k{hparams.top_k}_num_neighbors{hparams.num_neighbors}_dim{hparams.dim}_", suffix=".png")
        plt.savefig(tmp_png_file,transparent=True)
        plt.show()
        result = {   
    "origin_data": anydata, 
    "image":"",
    "table": [{"table_name": "Prompt Tokens and Their Top-K Similar Tokens", "table_list":all_tokens_info,"tabel_des": "", "tabel_res": ""}], 
    "result_des": "" 
    }   
        result["image"] = [{"image_name": f"{hparams.method}", "image_path": tmp_png_file, "image_des": "Dimension reduction results of embeddings of the original tokens and the top_k tokens with the largest similarity in vocabulary"}] 
     

    if dim == 3:
        fig = plt.figure(figsize=(14, 12))
       
        ax = fig.add_subplot(111, projection='3d') 
        ax.set_zlabel('Component 3')
        ax.set_ylabel('Component 2')
        ax.set_xlabel('Component 1')
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('0.6')
        ax.yaxis.pane.set_edgecolor('0.6')
        ax.zaxis.pane.set_edgecolor('0.6')
        ax.xaxis._axinfo['grid']['color'] = '0.6'
        ax.yaxis._axinfo['grid']['color'] = '0.6'
        ax.zaxis._axinfo['grid']['color'] = '0.6'
        all_tokens_info = []  
        for i, (orig_tok, _, topk_ids) in enumerate(original_tokens_positions):
            if i >= hparams.token_nums:
                break
            token_group_info = {"Original Token": orig_tok, "Similar Tokens": []}
            data_range_x = max(original_tokens_reduced[:, 0]) - min(original_tokens_reduced[:, 0])
            data_range_y = max(original_tokens_reduced[:, 1]) - min(original_tokens_reduced[:, 1])
            data_range_z = max(original_tokens_reduced[:, 2]) - min(original_tokens_reduced[:, 2])
            offset_factor_x = data_range_x * 0.003  
            offset_factor_y = data_range_y * 0.003
            offset_factor_z = data_range_z * 0.003
            if hparams.show_original_points == 1:
                ax.scatter(original_tokens_reduced[i, 0], original_tokens_reduced[i, 1], original_tokens_reduced[i, 2], color='blue', s=50)
            if hparams.show_original_tokens == 1:
                ax.text(original_tokens_reduced[i, 0] + offset_factor_x, original_tokens_reduced[i, 1] + offset_factor_y, original_tokens_reduced[i, 2] + offset_factor_z, orig_tok, fontsize=9, color='blue')
            for j in range(len(topk_ids)):
                idx = i * top_k + j
                similar_tok = tokenizer.convert_ids_to_tokens([topk_ids[j]])[0]
                token_group_info["Similar Tokens"].append(similar_tok)
                if hparams.show_connection_lines == 1:
                    ax.plot([original_tokens_reduced[i, 0], similar_tokens_reduced[idx, 0]],
                [original_tokens_reduced[i, 1], similar_tokens_reduced[idx, 1]],
                [original_tokens_reduced[i, 2], similar_tokens_reduced[idx, 2]], 'r--', alpha=0.3)
                if hparams.show_similar_points == 1:
                    ax.scatter(similar_tokens_reduced[idx, 0], similar_tokens_reduced[idx, 1], similar_tokens_reduced[idx, 2], color='red', alpha=0.7)
                if hparams.show_similar_tokens == 1:
                    ax.text(similar_tokens_reduced[idx, 0] + offset_factor_x, similar_tokens_reduced[idx, 1] + offset_factor_y, similar_tokens_reduced[idx, 2] + offset_factor_z,
                tokenizer.convert_ids_to_tokens([topk_ids[j]])[0], fontsize=8, color='red')
            all_tokens_info.append(token_group_info)
        ax.set_title(f"Visualization of Tokens and Their Top-K Similar Tokens with {hparams.method.upper()} to {dim}D")
        cell_text = []
        headers = ["Original Tokens", "Similar Tokens"]
        for group in all_tokens_info:
            #cell_text.append([group["Original Token"], ", ".join(group["Similar Tokens"])])
            cell_text.append([
    group["Original Token"].decode('utf-8') if isinstance(group["Original Token"], bytes) else group["Original Token"],
    " ".join([token.decode('utf-8') if isinstance(token, bytes) else token for token in group["Similar Tokens"]])])
        tmp_png_file = get_temp_file_with_prefix_suffix(prefix=f"method_{hparams.method}_token_nums{hparams.token_nums}_top_k{hparams.top_k}_num_neighbors{hparams.num_neighbors}_dim{hparams.dim}_", suffix=".png")
        plt.savefig(tmp_png_file,transparent=True)
        plt.show()
        result = {   
    "origin_data": anydata, 
    "image":"",
    "table": [{"table_name": "Prompt Tokens and Their Top-K Similar Tokens", "table_list":all_tokens_info,"tabel_des": "", "tabel_res": ""}], 
    "result_des": "" 
    }   
        result["image"] = [{"image_name": f"{hparams.method}", "image_path": tmp_png_file, "image_des": "Dimension reduction results of embeddings of the original tokens and the top_k tokens with the largest similarity in vocabulary"}] 
    
    return result