from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class KNHyperParams(HyperParams):
    lr_scale: float = None
    n_toks: int = None
    model_path: str = None
    refine: bool = None
    batch_size: int = None
    steps: int = None
    adaptive_threshold: float = None
    p: float = None

from .kn_main import diagnose
name = "KN"
requires_input_keys = ["prompts", "ground_truth"]
cost_seconds_per_query = 80
interpret_class=util.constant.neuron_attribution
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, util.constant.mlp, interpret_class]


