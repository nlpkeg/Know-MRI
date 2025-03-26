# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import einops
import collections
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *

class FiNEKnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None
        self.batch_activation = []

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate.dense"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif 'gptj' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.fc_in"
            self.output_ff_attr = "mlp.fc_out.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif "gpt2" == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'llama' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'baichuan' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif "t5" == model_type:
            self.transformer_layers_attr = "decoder.block"
            self.input_ff_attr = "layer.2.DenseReluDense.wi_0"
            self.output_ff_attr = "layer.2.DenseReluDense.wo.weight"
            self.word_embeddings_attr = "shared.weight"
        elif 'chatglm2' == model_type:
            self.transformer_layers_attr = "transformer.encoder.layers"
            self.input_ff_attr = "mlp.dense_4h_to_h"
            self.output_ff_attr = "mlp.dense_h_to_4h.weight"
            self.word_embeddings_attr = "transformer.embedding.word_embeddings.weight"
        elif 'internlm' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'qwen' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.w1"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'mistral' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        wtight =  get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )
        if "gpt2" == self.model_type:
            wtight = wtight.T
        return wtight

    def _get_input_ff_layer(self, layer_idx):
        wtight = get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr+".weight",
        )
        if "gpt2" == self.model_type:
            wtight = wtight.T
        return wtight

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            if self.model_type == "bert":
                prompt = prompt + self.tokenizer.mask_token
                encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            else:
                encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.model_type == 't5':
            target_input = self.tokenizer(target, return_tensors='pt').to(self.device)
            encoded_input['decoder_input_ids'] = target_input['input_ids']
            encoded_input['decoder_attention_mask'] = target_input['attention_mask']
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        elif self.model_type == 't5':
            mask_idx = list(range(encoded_input['decoder_input_ids'].size(1)))
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            #if "qwen" in self.model_type or "gpt" in self.model_type or 't5' in self.model_type or 'llama' in self.model_type:
            if "qwen" in self.model_type or "gpt" in self.model_type or 't5' in self.model_type or 'llama' in self.model_type or 'baichuan' in self.model_type or "chatglm" in self.model_type or "internlm" in self.model_type or "bert" in self.model_type:
                target = self.tokenizer.encode(target)
                te = target[0]
                if te == self.tokenizer.bos_token_id or te == self.tokenizer.unk_token_id:
                    target = target[1:]
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("qwen" in self.model_type or "gpt" in self.model_type or 'llama' in self.model_type) else 1
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i] if n_sampling_steps > 1 else target_label
            # gt_prob = probs[:, target_idx].item()
            if self.model_type == 't5':
                for q, target_idx_ in enumerate(target_idx):
                    gt_prob_= probs[:, q, target_idx_]
                    all_gt_probs.append(gt_prob_)

                    argmax_prob, argmax_id = [i.item() for i in probs[:,q,:].max(dim=-1)]
                    argmax_tokens.append(argmax_id)
                    argmax_str = self.tokenizer.decode([argmax_id])
                    all_argmax_probs.append(argmax_prob)

                    argmax_completion_str += argmax_str
            else:
                gt_prob = probs[:, target_idx]
                all_gt_probs.append(gt_prob)

                # get info about argmax completion
                argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
                argmax_tokens.append(argmax_id)
                argmax_str = self.tokenizer.decode([argmax_id])
                all_argmax_probs.append(argmax_prob)

                prompt += argmax_str
                argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4
        
    def clean_activation(self):
        self.batch_activation = []

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")

    def get_layers_activation(self, encoded_input: dict, layers: List[int], require_grad=False):

        def get_activations(model, layer_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """
            self.batch_activation = []

            def hook_fn(act):
                if self.model_type == 'chatglm2':
                    act = act.transpose(0, 1)
                self.batch_activation.append(act[0, :].detach().clone().cpu())

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )
        handdels = []
        for layer in layers:
            handdels.append(get_activations(self.model, layer_idx=layer))
        if require_grad:
            baseline_outputs = self.model(**encoded_input)
        else:
            with torch.no_grad():
                baseline_outputs = self.model(**encoded_input)
        for h in handdels:
            h.remove()
        batch_activation = self.batch_activation
        self.batch_activation = []
        return baseline_outputs, torch.stack(batch_activation)

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores_quick(self,
                         prompt: str,
                         ground_truth: str,
                         layers: List[int]):
        encoded_input, _, target_label = self._prepare_inputs(prompt+" "+ground_truth, ground_truth)
        target_len = len(target_label)
        model_output, activation = self.get_layers_activation(encoded_input=encoded_input, layers=layers)
        # num_layer * len * hidden_dim
        model_outputs_ids = model_output.logits.argmax(-1).cpu()
        original_output_ids = model_outputs_ids[0, -target_len - 1:-1] # 获取能够影响输出的activation
        activation = activation.permute(1, 0, 2)[-target_len - 1: -1, :]
        # target_len * num_layer * hidden_dim
        # scores = []
        scores = torch.zeros((activation.shape[1], activation.shape[2])).half().cuda()
        for j, id in enumerate(original_output_ids):
            _unembedding = self.model.get_output_embeddings()
            if _unembedding is None:
                unembedding = self._get_word_embeddings().T[:, id].detach().clone()
            else:
                unembedding = _unembedding.weight.T[:, id].detach().clone()
            for layer in layers:
                act = activation[j, layer].cuda()
                down_proj = self._get_output_ff_layer(layer_idx=layer).T.detach().clone()
                if 't5' in self.model_type:
                    act = act.float()
                    unembedding = unembedding.float()
                prob = act * ((down_proj.to(act.device) @ unembedding.to(act.device)).flatten())
                scores[layer] += prob.flatten()
        
        return scores.cpu()