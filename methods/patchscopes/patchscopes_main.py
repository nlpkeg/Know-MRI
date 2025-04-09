from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from pathlib import Path
from util.model_tokenizer import get_cached_model_tok
from . import patchscopesHyperParams
import numpy as np
from tqdm import tqdm
import torch
import transformers
import re
from ast import literal_eval
import functools
import json
import os
import random
import shutil
import pandas as pd
import datasets
import torch.nn as nn
import zipfile
from datasets import load_from_disk
from dataset_process import knowns, ZsRE, counterfact, pararel, PEP3k, TwentyQ

torch.set_grad_enabled(False)

tqdm.pandas()

def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )

def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p

###
def set_hs_patch_hooks_baichuan(
    model,
    hs_patch_config,
    module="hs",  
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """Baichuan patch hook"""
    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            #(hidden_states,)
            hidden_states = input[0]
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs

        def post_hook(module, input, output):
            # (hidden_states,)
            if "skip_ln" in name or "mlp" in name:
                hidden_states = output.clone()
            else:
                hidden_states = output[0].clone()
                
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return output
                
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs
                
            if "skip_ln" in name or "mlp" in name:
                return hidden_states
            return (hidden_states,) + output[1:]

        return pre_hook if patch_input else post_hook

    hooks = []
    for layer_idx in hs_patch_config:
        positions = hs_patch_config[layer_idx]
        hook_name = f"patch_{module}_{layer_idx}"

        if module == "hs":
            target_layer = model.model.layers[layer_idx]
        elif module == "mlp":
            target_layer = model.model.layers[layer_idx].mlp
        elif module == "attn":
            target_layer = model.model.layers[layer_idx].self_attn
        else:
            raise ValueError(f"Module %s not yet supported： {module}")

        if skip_final_ln and layer_idx == len(model.model.layers) - 1 and module == "hs":
            hook = model.model.norm.register_forward_hook(
                patch_hs(
                    f"{hook_name}_skip_ln",
                    positions,
                    patch_input,
                    generation_mode
                )
            )
        else:
            if patch_input:
                hook = target_layer.register_forward_pre_hook(
                    patch_hs(hook_name, positions, patch_input, generation_mode)
                )
            else:
                hook = target_layer.register_forward_hook(
                    patch_hs(hook_name, positions, patch_input, generation_mode)
                )
        
        hooks.append(hook)
    
    return hooks

def set_hs_patch_hooks_internlm(
    model,
    hs_patch_config,
    module="hs",
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """InternLM patch hooks."""
    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            hidden_states = input[0]
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs

        def post_hook(module, input, output):
            hidden_states = output[0].clone()
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return output
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs
            return (hidden_states,) + output[1:]
        
        return pre_hook if patch_input else post_hook

    hooks = []
    for layer_idx in hs_patch_config:
        positions = hs_patch_config[layer_idx]
        layer_module = model.model.layers[layer_idx]
        
        hook_fn = patch_hs(
            f"patch_hs_{layer_idx}",
            positions,
            patch_input,
            generation_mode
        )
        
        if patch_input:
            hook = layer_module.register_forward_pre_hook(hook_fn)
        else:
            hook = layer_module.register_forward_hook(hook_fn)
        hooks.append(hook)
    return hooks

def set_hs_patch_hooks_qwen(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """Qwen patch hooks."""
    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            hidden_states = input[0]
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs

        def post_hook(module, input, output):
            hidden_states = output[0].clone()
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return output
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs
            return (hidden_states,) + output[1:]
        
        return pre_hook if patch_input else post_hook

    hooks = []
    for layer_idx in hs_patch_config:
        positions = hs_patch_config[layer_idx]
        layer_module = model.transformer.h[layer_idx]
        hook_fn = patch_hs(
            f"patch_hs_{layer_idx}",
            positions,
            patch_input,
            generation_mode
        )
        if patch_input:
            hook = layer_module.register_forward_pre_hook(hook_fn)
        else:
            hook = layer_module.register_forward_hook(hook_fn)
        hooks.append(hook)
    return hooks

###

def set_hs_patch_hooks_gpt2(
    model,
    hs_patch_config,
    module="hs", 
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """gpt2 patch hooks."""
    if module != "hs":
        raise ValueError(f"Module %s not yet supported：{module}")

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            hidden_states = input[0]
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs 

        def post_hook(module, input, output):
            hidden_states = output[0].clone()  
            seq_len = hidden_states.size(1)
            if generation_mode and seq_len == 1:
                return output
            for pos, hs in position_hs:
                hidden_states[0, pos] = hs
            return (hidden_states,) + output[1:]

        return pre_hook if patch_input else post_hook

    hooks = []
    for layer_idx in hs_patch_config:
        positions = hs_patch_config[layer_idx]
        if skip_final_ln and layer_idx == len(model.transformer.h) - 1:
            hook_fn = patch_hs(
                f"patch_hs_{layer_idx}_skip_ln",
                positions,
                patch_input,
                generation_mode
            )
            hook = model.transformer.ln_f.register_forward_hook(hook_fn)
        else:
            layer_module = model.transformer.h[layer_idx]
            hook_fn = patch_hs(
                f"patch_hs_{layer_idx}",
                positions,
                patch_input,
                generation_mode
            )
            if patch_input:
                hook = layer_module.register_forward_pre_hook(hook_fn)
            else:
                hook = layer_module.register_forward_hook(hook_fn)
        hooks.append(hook)
    return hooks

def set_hs_patch_hooks_neox(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Neox patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    if patch_input:
      hooks.append(
          model.gpt_neox.layers[i].register_forward_pre_hook(
              patch_hs(
                  f"patch_hs_{i}",
                  hs_patch_config[i],
                  patch_input,
                  generation_mode,
              )
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the
      # same model, the final layer norm is not needed because it was already
      # applied (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.gpt_neox.layers) - 1:
        hooks.append(
            model.gpt_neox.final_layer_norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        hooks.append(
            model.gpt_neox.layers[i].register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )

  return hooks


def set_hs_patch_hooks_llama(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Llama patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name or "mlp" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name or "mlp" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    patch_hook = patch_hs(
        f"patch_{module}_{i}",
        position_hs=hs_patch_config[i],
        patch_input=patch_input,
        generation_mode=generation_mode,
    )
    if patch_input:
      if module == "hs":
        hooks.append(
            model.model.layers[i].register_forward_pre_hook(patch_hook)
        )
      elif module == "mlp":
        hooks.append(
            model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
        )
      elif module == "attn":
        hooks.append(
            model.model.layers[i].self_attn.register_forward_pre_hook(
                patch_hook
            )
        )
      else:
        raise ValueError("Module %s not supported", module)
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
        hooks.append(
            model.model.norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        if module == "hs":
          hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
        elif module == "mlp":
          hooks.append(
              model.model.layers[i].mlp.register_forward_hook(patch_hook)
          )
        elif module == "attn":
          hooks.append(
              model.model.layers[i].self_attn.register_forward_hook(patch_hook)
          )
        else:
          raise ValueError("Module %s not supported", module)

  return hooks


def set_hs_patch_hooks_gptj(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """GPTJ patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need
  # to handle these cases in this call because this hook wraps the generation
  # call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation
  # if we are handling the initial input or a future step and thus don't know
  # if a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    if patch_input:
      hooks.append(
          model.transformer.h[i].register_forward_pre_hook(
              patch_hs(
                  f"patch_hs_{i}",
                  hs_patch_config[i],
                  patch_input,
                  generation_mode,
              )
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.transformer.h) - 1:
        hooks.append(
            model.transformer.ln_f.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        hooks.append(
            model.transformer.h[i].register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )

  return hooks


def remove_hooks(hooks):
  for hook in hooks:
    hook.remove()



model_to_hook = {
    "gptj": set_hs_patch_hooks_gptj,
    "llama": set_hs_patch_hooks_llama,
    "gpt2": set_hs_patch_hooks_gpt2,
    "qwen": set_hs_patch_hooks_qwen,
    "internlm": set_hs_patch_hooks_internlm,
    "baichuan": set_hs_patch_hooks_baichuan,
    "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj
}


# Import statements and other code remain the same...

def train_and_save_mappings(
    mt,
    model_name_or_path,
    num_layers,
    prompt_target,
    trn_n=1000,
):
    base_dir = Path(__file__).parent.parent/"util"/"tmp"/"mapping_metrix_A"
    base_dir.mkdir(exist_ok=True)
    sanitized_model_name = re.sub(r'\W+', '_', model_name_or_path)
    
    pile_dataset = []
    
    dataset = knowns.KnownsDataset(loc=knowns.default_loc)
    for i in range(trn_n):
        sample = knowns.get_processed_kvs(dataset[i], ["prompt"])
        pile_dataset.append(sample["prompt"])
    
    pile_trn = pile_dataset[:trn_n]

    df_data = []
    for sentence in pile_trn:
        # Prepare inputs for the source sentence
        inp_source = make_inputs(mt.tokenizer, [sentence])
        position_source = len(inp_source['input_ids'][0]) - 1
        output_source = mt.model(**inp_source, output_hidden_states=True)
        hidden_source = output_source["hidden_states"]
        logits = output_source.logits[0, -1, :]
        answer_t = torch.argmax(logits, dim=0).item()
        # Prepare inputs for the target prompt
        inp_target = inp_source.copy()
        inp_target['input_ids'][0][-1] = answer_t
        position_target = len(inp_target["input_ids"][0]) - 1  # last token
        output_target = mt.model(**inp_target, output_hidden_states=True)
        hidden_target = output_target["hidden_states"]
        
        for layer in range(num_layers):
            # Extract hidden states and append to data
            hs_source = hidden_source[layer][0][position_source].detach().cpu().numpy()
            hs_target = hidden_target[layer][0][position_target].detach().cpu().numpy()
            df_data.append({'layer': layer, 'source': hs_source, 'target': hs_target})
    
    # Create DataFrame and train transformation matrices
    df_trn = pd.DataFrame(df_data)
    mappings = {}
    for layer in range(num_layers):
        X = np.array(df_trn[df_trn['layer'] == layer]['source'].tolist(), dtype=np.float32)
        Y = np.array(df_trn[df_trn['layer'] == layer]['target'].tolist(), dtype=np.float32)
        # Pad and unpad 

        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        A, _, _, _ = np.linalg.lstsq(pad(X), pad(Y), rcond=None)
        mappings[layer] = A
        mapping_file = os.path.join(base_dir, f'{sanitized_model_name}_mapping_layer{layer}.npy')
        np.save(mapping_file, A.astype(np.float16))


def evaluate_patch_next_token_prediction(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    position_prediction=-1,
    transform=None,
):
  """Evaluate next token prediction."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt.tokenizer, [prompt_target])
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on without patching and get the results.
  inp_source = make_inputs(mt.tokenizer, [prompt_source])
  output_orig = mt.model(**inp_source, output_hidden_states=True)
  dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
  _, answer_t_orig = torch.max(dist_orig, dim=0)
  hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
      position_source
  ]
  if hidden_rep.dim() == 1:
    hidden_rep = hidden_rep.unsqueeze(0)
  if transform is not None:
    hidden_rep = transform(hidden_rep)
    hidden_rep = hidden_rep.squeeze() 

  # now do a second run on prompt, while patching the input hidden state.
  hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
  if layer_source == layer_target == mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False
  patch_hooks = mt.set_hs_patch_hooks(
      mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  output = mt.model(**inp_target)
  dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
  _, answer_t = torch.max(dist, dim=0)

  # remove patching hooks
  remove_hooks(patch_hooks)

  return answer_t



lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    

    result = dict()
    result["output"] = sample["ground_truth"]
    result["origin_data"] = []
    result["table"] = []

    with lock_kn:
        # Method preparation
        hparams = patchscopesHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        
        need_mapping = hparams.need_mapping
        print("Apply mapping =",need_mapping)
        
        mt = get_cached_model_tok(model_name=hparams.model_path)
        mt.set_hs_patch_hooks = model_to_hook[mt.model_type]
        mt.model.eval()
        
        # Initialize prompt_target and position_target
        prompt_target = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
        position_target = -1  # Last position
        
        base_dir = Path(__file__).parent.parent/"util"/"tmp"/"mapping_metrix_A"

        sanitized_model_name = re.sub(r'\W+', '_', model_name_or_path)
        
        # Determine number of layers
        num_layers = mt.num_layers

        if need_mapping:
            # Check if all mapping files exist
            all_mappings_exist = True
            for layer in range(num_layers):
                mapping_file = os.path.join(base_dir, f'{sanitized_model_name}_mapping_layer{layer}.npy')
                if not os.path.exists(mapping_file):
                    all_mappings_exist = False
                    break
            
            if not all_mappings_exist:
                print(f"start training mapping matrix A, num_layers: {num_layers}")
                train_and_save_mappings(mt, model_name_or_path, num_layers, prompt_target)
            else:
                print(f"Mapping matrix A already exist, num_layers: {num_layers}")
            
            # Load pre-trained transformation matrices
            mappings = {}
            for layer in range(num_layers):
                mapping_file = os.path.join(base_dir, f'{sanitized_model_name}_mapping_layer{layer}.npy')
                if os.path.exists(mapping_file):
                    A = np.load(mapping_file)
                    mappings[layer] = A
                else:
                    mappings[layer] = None  # Handle missing mappings as needed
        else:
            # If A is False, do not load or train mappings
            mappings = None


        # Collect table rows and build res summary
        rows = []
        res_string = " LLMs can be used to explain their own hidden layer representations through a framework called Patchscope. Predicted Next Tokens by Layer:\n"
        
        origin_data_list = []
        for layer_source in range(num_layers - 1):
            # Determine layer_target as the last layer
            layer_target = layer_source

            # Determine position_source as the last position of prompt_source
            prompt_source = sample["prompt"]
            inp_source = make_inputs(mt.tokenizer, [prompt_source])
            position_source = len(inp_source["input_ids"][0]) - 1  # Last position

            # Load transformation matrix A for this layer
            if need_mapping and mappings:
                # Load transformation matrix A for this layer
                A = mappings.get(layer_source, None)
                if A is not None:
                    # Define transformation function using PyTorch
                    A_tensor = torch.from_numpy(A).to(mt.model.device).to(mt.model.dtype)
                    # Pad and unpad 
                    pad = lambda x: torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
                    unpad = lambda x: x[:, :-1]
                    transform = lambda x: unpad(pad(x) @ A_tensor)
                else:
                    transform = None
            else:
                transform = None  # No transformation if A is False or mappings not available

            # Evaluate next token prediction for this layer_source
            answer_t = evaluate_patch_next_token_prediction(
                mt,
                prompt_source,
                prompt_target,
                layer_source,
                layer_target,
                position_source,
                position_target,
                module="hs",
                position_prediction=position_target,
                transform=transform  # Use the transformation here
            )

            # Decode the predicted token
            predicted_token = mt.tokenizer.decode([answer_t])
            
            origin_data_list.append({
                "Layer name": f"Layer_{layer_source}",
                "predicted_token": predicted_token
            })

            # Collect the result for this layer_source
            rows.append({
                "Layer name": f"Layer_{layer_source}",
                "Next Token Predicted by Model": predicted_token
            })
            if layer_source >= int(0.8 * num_layers):
              res_string += f"Layer {layer_source}: {predicted_token}\n"

        # Create table data structure
        table_data = {
            "table_name": "Predicted Next Tokens by Layer Source",
            "table_des": "This table shows the predicted next tokens for each layer source.",
            "table_list": rows,
            "table_res": res_string.strip()
        }

        # Assign to result
        result["origin_data"] = origin_data_list
        result["table"] = [table_data]
        result["result_des"] = ""

    return result
