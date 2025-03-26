from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class AttentionWeightsHyperParams(HyperParams):
    model_path: str = None
    num_heads: int = None

from .attention_weights_main import diagnose
name = "Attention Weights"
requires_input_keys = ["prompt"]
cost_seconds_per_query = 0.9
interpret_class=util.constant.attention
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, interpret_class]

