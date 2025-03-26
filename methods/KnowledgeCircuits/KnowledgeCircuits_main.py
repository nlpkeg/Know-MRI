from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.model_tokenizer import get_cached_model_tok
from . import KnowledgeCircuitsHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
import numpy as np
from tqdm import tqdm
import torch
from .transformer_lens import HookedTransformer
from functools import partial
import torch.nn.functional as F

from .eap.metrics import logit_diff, direct_logit
from .eap.graph import Graph
from .eap.attribute import attribute
import time
from rich import print as rprint
import pandas as pd
import re
import ast
from pathlib import Path
import json
from openai import OpenAI
import matplotlib.pyplot as plt
from IPython.display import display
temp_dir = Path(__file__).parent.parent.parent/"util"/"tmp"
temp_dir.mkdir(exist_ok=True)
    
lock_kn = threading.Lock()


def generate_corrupted_subject(clean_subject: str, relation: str, clean_label: str, tokenizer, api_key, base_url) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    excluded_subjects = []  
    max_attempts = 50       
    attempt = 0

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": f"Gnerate a subject in the same type with {clean_subject}, But not {relation} {clean_label}, Just output the generated word!"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        max_tokens=10,
    )

    corrupted_subject = response.choices[0].message.content
    excluded_subjects.append(corrupted_subject)

    if '{}' in relation:
            clean = relation.format(clean_subject)
            corrupted = relation.format(corrupted_subject)
    else:
            clean = f"{clean_subject} {relation}"
            corrupted = f"{corrupted_subject} {relation}"
    
    clean_tokens = tokenizer.tokenize(clean)
    corrupted_tokens = tokenizer.tokenize(corrupted)
    clean_tokens_len = len(tokenizer.tokenize(clean))
    while len(corrupted_tokens) != len(clean_tokens) and attempt < max_attempts: 
            exclude_prompt = ""
            length_hint = ""
        
            if excluded_subjects:
               exclude_prompt = f" Avoid these examples: {', '.join(excluded_subjects)}."
            
            if excluded_subjects:
                last_subject = excluded_subjects[-1]
                if '{}' in relation:
                    last = relation.format(last_subject)
                else:
                    last = f"{last_subject} {relation}"
                last_len = len(tokenizer.tokenize(last))
                if last_len < clean_tokens_len:
                    length_hint = f" Generate a longer word than {last_subject}."
                elif last_len > clean_tokens_len:
                    length_hint = f" Generate a shorter word than {last_subject}."
        
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"Generate a subject in the same type with {clean_subject}, "
                                          f"but not {relation} {clean_label}.{exclude_prompt}{length_hint} "
                                          "Just output the generated word!"}
            ]
        
            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=messages,
                max_tokens=10,
            )
        
            corrupted_subject = response.choices[0].message.content.strip()
            if '{}' in relation:
                corrupted = relation.format(corrupted_subject)
            else:
                corrupted = f"{corrupted_subject} {relation}"
        
            corrupted_tokens_len = len(tokenizer.tokenize(corrupted))
        
            if corrupted_tokens_len == clean_tokens_len:
                return corrupted_subject
        
            excluded_subjects.append(corrupted_subject)
            attempt += 1
    print(f"totally attempt:{attempt}")
    return excluded_subjects[-1] if excluded_subjects else ""



def generate_corrupted_label(clean_label: str, api_key, base_url) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": f" Only generate a word in the same type with {clean_label} but not exactl the same, Just output the generated word!"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        max_tokens=10,
    )

    corrupted_label = response.choices[0].message.content

    return corrupted_label



def diagnose(sample, model_name_or_path, hparams=None):
    result = dict()
    result["output"] = sample["ground_truth"]
    result["image"] = []
    
    with lock_kn:
        # method prepare
        hparams = KnowledgeCircuitsHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok(model_name=hparams.model_path)
        model=mt.model
        
        n = torch.cuda.device_count()
        if hparams.model_name == "openai-community/gpt2-xl":
            model_name = "gpt2-xl"
        else:
            model_name = hparams.model_name
        model = HookedTransformer.from_pretrained(model_name=model_name, hf_model=model, device="cuda", n_devices=n, move_to_device=True, fold_ln=False, center_writing_weights=False, center_unembed=False)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True


        
        #clean_subject = 'Eiffel Tower'
        #corrupted_subject = 'the Great Walls'
        #clean = f'The official currency of the country where {clean_subject} is loacted in is the'
        #corrupted = f'The official currency of the country where {corrupted_subject} is loacted in is the'
        #labels = ['Euro','Chinese']


        clean_subject = sample["triple_subject"]
        print(f"clean_subject  (before adjustment): {clean_subject}")
        
        relation = sample["triple_relation"]
        print(f"relation: {relation}")    
        
        clean_label = sample["triple_object"]
        print(f"clean_label: {clean_label}") 

        corrupted_subject = generate_corrupted_subject(clean_subject, relation, clean_label, mt.tokenizer, hparams.api_key, hparams.base_url)
        print(f"corrupted_subject (before adjustment): {corrupted_subject}")
        
        corrupted_label = generate_corrupted_label(clean_label, hparams.api_key, hparams.base_url)
        print(f"corrupted_label: {corrupted_label}")
        
        if '{}' in relation:
            clean = relation.format(clean_subject)
            corrupted = relation.format(corrupted_subject)
        else:
            clean = f"{clean_subject} {relation}"
            corrupted = f"{corrupted_subject} {relation}"
        
        print(f"clean (subject+relation): {clean}") 
        print(f"corrupted (subject+relation): {corrupted}")
        
        labels = [clean_label,corrupted_label]


        label_idx = model.tokenizer(labels[0],add_special_tokens=False).input_ids[0]
        corrupted_label_idx = model.tokenizer(labels[1],add_special_tokens=False).input_ids[0]
        label = [[label_idx, corrupted_label_idx]]
        label = torch.tensor(label)
        data = ([clean],[corrupted],label)
        
        g = Graph.from_model(model)
        with torch.set_grad_enabled(True):
            attribute(model, g, data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-case', ig_steps=100)
        g.apply_topn(hparams.topn, absolute=True)
        g.prune_dead_nodes()
       
        # graph_json_path = temp_dir / "graph.json"
        graph_img_path = get_temp_file_with_prefix_suffix(suffix=".png")
        graph_json_path = temp_dir / "graph.json"
       
        g.to_json(graph_json_path)
        gz = g.to_graphviz()
        gz.draw(graph_img_path, prog='dot')
        #show image
        plt.imshow(plt.imread(graph_img_path))
        display(plt.show())
        #
        result["image"].append({"image_name": "Knowledge Circuit Graph",
                                "image_path": graph_img_path,
                                "image_des": "The Knowledge Circuit Graph is used to visualize the key nodes and edges when a model processes a specific task. In the graph, nodes represent different computing units in the model (such as attention heads and MLP nodes), and edges represent the connections between these nodes and their importance (indicated by the thickness of the edges). The thickness reflects the weight or score of the edges, helping to identify the most important paths and connections in the model. When connected to attention heads: Edges of the q type will be purple. Edges of the k type will be green. Edges of the v type will be blue. Such color differentiation is used to visualize different parts of the attention mechanism. When connected to MLP nodes: Colors are used to represent the difference between positive and negative weights. Red represents negative weights, while black represents positive weights.",
                                "image_res": "The graph displays the key nodes and edges in the model when processing input data. After filtering operations, only the edges with the highest scores are retained.",
                              })
        with open(graph_json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        result["origin_data"] = {"Important Nodes and Edges for Knowledge Circuit Graph": graph_data}

        return result
        
        