# helper functions for patching torch transformer models
import torch
import torch.nn as nn
import collections
from typing import List, Callable
import torch
import torch.nn.functional as F
import collections


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)


def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Gets the feedforward layer of a model within the transformer block
    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer


def register_hook(
    model: nn.Module,
    layer_idx: int,
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Registers a forward hook in a pytorch transformer model that applies some function, f, to the intermediate
    activations of the transformer model.

    specify how to access the transformer layers (which are expected to be indexable - i.e a ModuleList) with transformer_layers_attr
    and how to access the ff layer with ff_attrs

    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `f`: Callable
      a callable function that takes in the intermediate activations
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    ff_layer = get_ff_layer(
        model,
        layer_idx,
        transformer_layers_attr=transformer_layers_attr,
        ff_attrs=ff_attrs,
    )

    def hook_fn(m, i, o):
        f(o)

    return ff_layer.register_forward_hook(hook_fn)


class Patch(torch.nn.Module):
    """
    Patches a torch module to replace/suppress/enhance the intermediate activations
    """

    def __init__(
        self,
        ff_layer: nn.Linear,
        mask_idx: int,
        replacement_activations: torch.Tensor = None,
        target_positions: List[List[int]] = None,
        mode: str = "replace",
        enhance_value: float = 2.0,
    ):
        super().__init__()
        self.ff = ff_layer
        # nn.Linear()
        self.acts = replacement_activations
        self.mask_idx = mask_idx
        self.target_positions = target_positions
        self.enhance_value = enhance_value
        assert mode in ["replace", "suppress", "enhance", "FT"]
        self.mode = mode
        if self.mode == "replace":
            assert self.acts is not None
        elif self.mode in ["enhance", "suppress"]:
            assert self.target_positions is not None
        elif self.mode == "FT":
            a = self.ff.weight
            self.ff.weight.requires_grad = False
            self.delta_neurons = []
            for pos in self.target_positions:
                li = torch.nn.Linear(in_features=self.ff.in_features, out_features=1,dtype=self.ff.weight.dtype, device=self.ff.weight.device, bias=False)
                torch.nn.init.constant_(li.weight, 0)
                self.delta_neurons.append(li)

            

    def forward(self, inp: torch.Tensor):
        # x: batch_szie * length * hidden_dim
        x = self.ff(inp)
        if self.mode == "replace":
            x[:, self.mask_idx, :] = self.acts
        elif self.mode == "suppress":
            for pos in self.target_positions:
                x[:, self.mask_idx, pos] = 0.0
        elif self.mode == "enhance":
            for pos in self.target_positions:
                x[:, self.mask_idx, pos] *= self.enhance_value
        elif self.mode == "FT":
            for ind, pos in enumerate(self.target_positions):
                tem = self.delta_neurons[ind](inp)
                # batch_szie * length * 1
                # te = x[:, :, pos]
                x[:, :, pos] += torch.squeeze(tem, dim=-1) 
        else:
            raise NotImplementedError
        return x


def patch_ff_layer(
    model: nn.Module,
    mask_idx: int,
    layer_idx: int = None,
    replacement_activations: torch.Tensor = None,
    mode: str = "replace",
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate", # self.input_ff_attr
    neurons: List[List[int]] = None,
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    `mask_index` with `replacement_activations`

    `model`: nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
      which transformer layer to access
    `mask_idx`: int
      the index (along the sequence length) of the activation to replace.
      TODO: multiple indices
    `replacement_activations`: torch.Tensor
      activations [taken from the mask_idx position of the unmodified activations] of shape [b, d]
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)

    if mode == "replace":
        ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
        assert layer_idx < len(
            transformer_layers
        ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"

        set_attribute_recursive(
            transformer_layers[layer_idx],
            ff_attrs,
            Patch(
                ff_layer,
                mask_idx,
                replacement_activations=replacement_activations,
                mode=mode,
            ),
        )
        return None

    elif mode in ["suppress", "enhance"]:
        neurons_dict = collections.defaultdict(list)
        for neuron in neurons:
            layer_idx, pos = neuron
            neurons_dict[layer_idx].append(pos)
        for layer_idx, positions in neurons_dict.items():
            # positions当前层神经元的id
            assert layer_idx < len(transformer_layers)
            ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
            set_attribute_recursive(
                transformer_layers[layer_idx],
                ff_attrs,
                Patch(
                    ff_layer,
                    mask_idx,
                    replacement_activations=None,
                    mode=mode,
                    target_positions=positions,
                ),
            )
        return None
    elif mode == "FT":
        neurons_dict = collections.defaultdict(list)
        li = []
        for neuron in neurons:
            layer_idx, pos = neuron
            neurons_dict[layer_idx].append(pos)
        for layer_idx, positions in neurons_dict.items():
            assert layer_idx < len(transformer_layers)
            ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
            tem = Patch(ff_layer, mask_idx, replacement_activations=None, mode=mode, target_positions=positions)
            set_attribute_recursive(
                transformer_layers[layer_idx],
                ff_attrs,
                tem
            )
            li.append(tem)
        return li
    else:
        raise NotImplementedError


def unpatch_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
    # add: bool = False,
    neuron_id = None,
    mode="FT",
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.

    `model`: torch.nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    # weight: out_feature * in_faeture
    if mode=="FT":
        with torch.no_grad():
          for i, ind in enumerate(ff_layer.target_positions):
              ff_layer.ff.weight[ind, :] += torch.squeeze(ff_layer.delta_neurons[i].weight).to(ff_layer.ff.weight.device)
        assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
        set_attribute_recursive(
        transformer_layers[layer_idx],
        ff_attrs,
        ff_layer.ff,)

    elif mode=="erase":
        with torch.no_grad():
          for ind in neuron_id:
              ff_layer.weight[ind, :] *= 0
        # return None
    elif mode=="enhance":
         with torch.no_grad():
          for ind in neuron_id:
              ff_layer.weight[ind, :] *= 2
    else:
        assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
        set_attribute_recursive(
        transformer_layers[layer_idx],
        ff_attrs,
        ff_layer.ff,)

    


def unpatch_ff_layers(
    model: nn.Module,
    layer_indices: List,
    neurons: List,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
    mode="FT"
):
    """
    Calls unpatch_ff_layer for all layers in layer_indices
    """
    for layer_idx in layer_indices:
        neuron_id = [te[1] for te in neurons if te[0]==layer_idx]
        unpatch_ff_layer(model, layer_idx, transformer_layers_attr, ff_attrs, mode=mode, neuron_id=neuron_id)
