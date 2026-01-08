import os
import json
import torch
import numpy as np
import gc
from tqdm import tqdm
from transformers import AutoTokenizer 

from diagnose import diagnose
from models import llama, gptj
from methods import causal_trace
from EasyEdit.easyeditor import BaseEditor, MEMITHyperParams
from EasyEdit.easyeditor.models.memit.memit_main import COV_CACHE
import traceback
import argparse
import yaml
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def parse_args():
    parser = argparse.ArgumentParser(description="Model editing with MEMIT method")
    
    # 数据相关参数
    parser.add_argument("--data_type", type=str, default="counterfact", help="Dataset type (default: counterfact)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="EleutherAI/gpt-j-6b", help="Path to the model")
    parser.add_argument("--hparams_path", type=str, default="./evaluation/gpt-j-6B.yaml", help="Path to hyperparameters YAML file")
    parser.add_argument("--output_dir", type=str, default="./evaluation/result", help="Directory to save output results")
    
    # 预计算缓存相关参数
    parser.add_argument("--pre_calculated_file", type=str, default="", help="Path to pre-calculated metrics file")
    
    # 其他参数
    parser.add_argument("--topk", type=int, default=5, help="Top-k layers to consider (default: 5)")
    
    return parser.parse_args()

args = parse_args()

OUTPUT_DIR = args.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. 定义文件路径 ===
# 当前任务的输出结果文件
DATA_TYPE = args.data_type
RESULTS_FILE = os.path.join(OUTPUT_DIR, f"final_edit_metrics_gpt_{DATA_TYPE}.json")
# 包含预计算 indices 的参考文件
PRE_CALCULATED_METRICS_FILE = args.pre_calculated_file

# Hparams 路径
HPARAMS_PATH = args.hparams_path
MODEL_PATH = args.model_path 

# 数据路径
DATASET_PATH = f"./evaluation/dataset/{DATA_TYPE}.json"

topk = args.topk

print(f"Swithing model path in config:")
with open(HPARAMS_PATH, 'r') as f:
    hparams_config = yaml.safe_load(f)

raw_model_name = hparams_config.get('model_name', 'unknown_model')
hparams_config['model_name'] = MODEL_PATH
with open(HPARAMS_PATH.replace(".yaml", "_copy.yaml"), 'w') as f:
    yaml.dump(hparams_config, f)
HPARAMS_PATH = HPARAMS_PATH.replace(".yaml", "_copy.yaml")

dig_path = f"./methods/causal_trace/hparams/{raw_model_name.replace('/', '_')}.json"
with open(dig_path, 'r') as f:
    ct_hparams = json.load(f)
ct_hparams["model_path"] = MODEL_PATH
with open(dig_path, 'w') as f:
    json.dump(ct_hparams, f, indent=4)

print("Loading Dataset...")
with open(DATASET_PATH, "r", encoding='utf-8') as f:
    dataset = json.load(f)

if args.num_samples is not None:
    dataset = dataset[:args.num_samples]


# === 2. 加载输出结果（用于断点续传） ===
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding='utf-8') as f:
        final_results_data = json.load(f)
    processed_indices = {entry["index"] for entry in final_results_data}
    print(f"Loaded {len(final_results_data)} previously processed samples (Output File)")
else:
    final_results_data = []
    processed_indices = set()

# === 3. 加载预计算的 Indices 缓存 ===
pre_calculated_cache = {}
if os.path.exists(PRE_CALCULATED_METRICS_FILE):
    print(f"Loading pre-calculated metrics from: {PRE_CALCULATED_METRICS_FILE}")
    with open(PRE_CALCULATED_METRICS_FILE, "r", encoding='utf-8') as f:
        pre_calc_data = json.load(f)
        for item in pre_calc_data:
            # 建立映射: index -> {max_token_idx, top_layer_indices}
            if "index" in item and "max_token_idx" in item and "top_layer_indices" in item:
                pre_calculated_cache[item["index"]] = {
                    "max_token_idx": item["max_token_idx"],
                    "top_layer_indices": item["top_layer_indices"],
                    "subject": item.get("subject")
                }
    print(f"Loaded {len(pre_calculated_cache)} pre-calculated indices.")
else:
    print("Warning: Pre-calculated metrics file not found. Will run diagnose for all.")

hparams = MEMITHyperParams.from_hparams(HPARAMS_PATH)

print("Starting Processing Loop...")
for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    
    if i in processed_indices:
        print(f"[Index {i}] Already processed. Skipping.")
        continue
    subject = data['subject']
    prompt_input = data['origin_question']
    gt_old = data['origin_answer']
    prompt_choice = data['question']
    gt_choice = data['answer'] 
    target_new_fact = gt_choice

    sample = {
        'subject': subject,
        'prompt': prompt_input,
        'ground_truth': gt_old,
        'target_new': target_new_fact, 
        'triple_subject': subject,
        'prompt_choice': prompt_choice,
        'gt_choice': gt_choice    
    }
    if i == 0:
        print(sample)

    # === 4. 获取 Indices (优先从缓存读取，否则运行 diagnose) ===
    max_token_idx = None
    top_layer_list = None

    if i in pre_calculated_cache:
        # [命中缓存] 直接读取
        cached_info = pre_calculated_cache[i]
        max_token_idx = cached_info["max_token_idx"]
        top_layer_list = cached_info["top_layer_indices"]
        if cached_info.get("subject"):
            subject = cached_info["subject"]
        # print(f"[Index {i}] Using pre-calculated indices.") # 可选：打印日志
    else:
        # [未命中缓存] 运行 diagnose
        try:
            torch.cuda.empty_cache()
            print(f"[Index {i}] Pre-calculated data not found. Running Diagnose...")
            
            result = diagnose.diagnosing(
                sample=sample, 
                model_name_or_path=gptj, 
                method=causal_trace.name
            )
            scores = np.array(result["origin_data"]["Restoring state score"])
            
            # 计算最显著的 token 和 layer
            max_flat_index = np.argmax(scores)
            calc_max_token_idx, max_layer_idx_of_max_val = np.unravel_index(max_flat_index, scores.shape)
            target_token_layer_scores = scores[calc_max_token_idx]
            
            # 获取 topk layer
            top_layer_indices = np.argsort(target_token_layer_scores)[-topk:][::-1]
            
            max_token_idx = int(calc_max_token_idx)
            top_layer_list = top_layer_indices.tolist()

            del result
            del scores
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n[Index {i}] Diagnose failed: {e}")
            continue

    # 再次检查是否成功获取到了必要参数
    if max_token_idx is None or top_layer_list is None:
        print(f"[Index {i}] Failed to get token/layer indices. Skipping.")
        continue

    # === 5. 执行编辑 (EasyEdit) ===
    hparams.layers = [int(x) for x in top_layer_list]
    # hparams.fact_token = f"index_{int(max_token_idx)}"
    
    editor = BaseEditor.from_hparams(hparams)
    
    try:
        with torch.enable_grad():
            metrics, edited_model, _ = editor.edit(
                prompts=sample["prompt_choice"],          
                ground_truth=sample["gt_choice"], 
                target_new=sample["target_new"],     
                subject=sample["subject"],         
                sequential_edit=True,
                keep_original_weight=False        
            )
        
        # === 6. 保存单条结果 ===
        result_entry = {
            "index": i,
            "prompt": sample["prompt_choice"],
            "subject": sample["subject"],
            "ground_truth": sample["gt_choice"],
            "target_new": sample["target_new"],
            "max_token_idx": int(max_token_idx),
            "top_layer_indices": top_layer_list,
            "edit_metrics": metrics 
        }
        
        final_results_data.append(result_entry)
        
        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
            json.dump(final_results_data, f, indent=4, ensure_ascii=False)
            
        print(f"[Index {i}] Successfully processed (Indices source: {'Cache' if i in pre_calculated_cache else 'Diagnose'})")

    except Exception as e:
        print(f"[Index {i}] Edit failed: {e}")
        print(traceback.format_exc())

    # === 资源清理 ===
    if editor is not None:
        del editor
    if 'edited_model' in locals() and edited_model is not None:
        del edited_model
    gc.collect()
    torch.cuda.empty_cache()
    COV_CACHE.clear() 
    try:
        torch.cuda.ipc_collect()
    except:
        pass

print("Processing completed.")