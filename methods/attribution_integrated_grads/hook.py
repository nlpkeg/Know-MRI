from util.model_tokenizer import get_cached_model_tok, ModelAndTokenizer
import collections
import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

def get_attributes(x: torch.nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def register_hook(
    model: torch.nn.Module,
    f: Callable,
    embedding_attrs: str = "intermediate",
):
    emb_layer = get_attributes(model, embedding_attrs)

    def hook_fn(m, i, o):
        f(o)

    return emb_layer.register_forward_hook(hook_fn)


class Patch(torch.nn.Module):
    """
    Patches a torch module to replace/suppress/enhance the intermediate activations
    """

    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        prompt_length: int,
        replacement_activations: torch.Tensor = None
    ):
        super().__init__()
        self.emb = embedding_layer
        self.acts = replacement_activations
        self.prompt_length = prompt_length

    def forward(self, x: torch.Tensor):
        x = self.ff(x)
        x[:, :self.prompt_length, :] = self.acts
        return x

def set_attribute_recursive(x: torch.nn.Module, attributes: "str", new_attribute: torch.nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)

def patch_emb_layer(
    mt: ModelAndTokenizer,
    prompt_length: int,
    replacement_activations: torch.Tensor = None):

    emb_layer = get_attributes(mt.model, mt.word_embeddings_attr.replace(".weight", ""))

    setattr(mt.model, mt.word_embeddings_attr.replace(".weight", ""), 
            Patch(embedding_layer=emb_layer, prompt_length=prompt_length, replacement_activations=replacement_activations))

def unpatch_emb_layer(mt: ModelAndTokenizer):

    emb_layer = get_attributes(mt.model, mt.word_embeddings_attr.replace(".weight", ""))

    setattr(mt.model, mt.word_embeddings_attr.replace(".weight", ""), 
            emb_layer.emb)



class Attribution:
    def __init__(
        self,
        mt: ModelAndTokenizer,
        model_type: str = "bert",
        device: str = None,
    ):
        self.mt = mt
        self.model = self.mt.model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)
        self.tokenizer = self.mt.tokenizer
        self.score = None

        self.baseline_activations = None

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
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
            self.input_ff_attr = "layer.2.DenseReluDense.wi"
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
        elif 'qwen2' == model_type:
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

        self.baseline_activations = None

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
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
            if "qwen" in self.model_type or "gpt" in self.model_type or 't5' in self.model_type or 'llama' in self.model_type:
                target = self.tokenizer.encode(target, add_special_tokens=False)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target, add_special_tokens=False)
        return encoded_input, mask_idx, target
    
    def get_baseline_with_activations(self, encoded_input: dict, prompt_length: int):
        def get_activations(model, prompt_length):
            def hook_fn(acts):
                self.baseline_activations = acts[:, :prompt_length, :]

            return register_hook(model, f=hook_fn, embedding_attrs=self.mt.word_embeddings_attr.replace(".weight", ""))
        handle = get_activations(self.mt.model, prompt_length)
        with torch.no_grad():
            baseline_outputs = self.mt.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations
    
    def modify_activations(self, encoded_input: dict, prompt_length: int, activations):
        def mod_activations(model, prompt_length):
            def hook_fn(acts):
                acts[:, :prompt_length, :] = activations

            return register_hook(model, f=hook_fn, embedding_attrs=self.mt.word_embeddings_attr.replace(".weight", ""))
        handle = mod_activations(self.mt.model, prompt_length)
        baseline_outputs = self.mt.model(**encoded_input)
        handle.remove()
        return baseline_outputs
    

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
            # print(probs.shape)
            # gt_prob = probs[:, target_idx].item()
            # print(target_idx)
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
                # print(gt_prob.shape)
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
        steps = steps + 1
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            acts = (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
            return acts[1:]
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")

    def get_attribution_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        if "qwen" in self.mt.model_type or "gpt" in self.mt.model_type or 't5' in self.mt.model_type or 'llama' in self.mt.model_type:
            prompt_list = self.mt.tokenizer.encode(prompt)
            te = self.mt.tokenizer.encode(ground_truth, add_special_tokens = False)
        else:
            prompt_list = self.mt.tokenizer.convert_tokens_to_ids(prompt)
            te = self.mt.tokenizer.convert_tokens_to_ids(ground_truth, add_special_tokens = False)

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("qwen" in self.model_type or "gpt" in self.model_type or 'llama' in self.model_type) else 1
        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0 and ("qwen" in self.model_type or self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input=encoded_input, prompt_length=len(prompt_list) 
                )
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                    if self.model_type == 't5':
                        inputs["decoder_input_ids"] = einops.repeat(
                            encoded_input["decoder_input_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                        inputs["decoder_attention_mask"] = einops.repeat(
                            encoded_input["decoder_attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations

                    # then forward through the model to get the logits
                    outputs = self.modify_activations(encoded_input=inputs, prompt_length=len(prompt_list), activations=batch_weights)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    target_idx = (
                        target_label[i] if n_sampling_steps > 1 else target_label
                    )
                    if self.model_type == 't5':
                        assert probs.size(1) == len(target_idx)
                        target_probs = [probs[:, q, target_idx_] for q, target_idx_ in enumerate(target_idx)]

                        grad = torch.autograd.grad(
                            torch.unbind(torch.cat(target_probs, dim=0)), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)
                    # elif self.model_type == 'chatglm2':
                    #     grads = [torch.autograd.grad(torch.sum(prob), batch_weights)[0] for prob in torch.unbind(probs[:, target_idx])]
                    #     grad = torch.stack(grads).sum(dim=0)
                    #     integrated_grads_this_step.append(grad)
                    else:
                        grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_idx]), batch_weights
                        )[0]
                        # grad = grad.view(len(prompt_list), -1)
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                
                if self.model_type == "chatglm2":
                    baseline_activations = baseline_activations.mean(dim=0)
                    # baseline_activations = baseline_activations[1]
                    integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                else:
                    integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step.sum(dim=-1))

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).cpu().tolist()
            return integrated_grads, prompt_list, te
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and ("qwen" in self.model_type or self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input=encoded_input, prompt_length=len(prompt_list) 
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError