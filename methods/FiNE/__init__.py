from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class FiNEHyperParams(HyperParams):
    max_layer: int = None
    model_path: str = None
    refine: bool = None
    adaptive_threshold: float = None
    num_neuron: int = None
    unembedding_num: int = None
    p: float = None

from .FiNE_main import diagnose
name = "FiNE"
requires_input_keys = ["prompt", "ground_truth"]
cost_seconds_per_query = 1
interpret_class=util.constant.neuron_attribution
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, util.constant.mlp, interpret_class]

