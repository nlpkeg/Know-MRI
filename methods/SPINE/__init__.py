from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class SPINEHyperParams(HyperParams):
    model_path: str = None
    top_k: int = None
    epoch: int = None
    lr: float = None
    asl: float = None
    psl: float = None
    hidden_dim: int = None
    noise: float = None
    mean_value: float = None
    topk_tokens: int = None
  

from .SPINE_main import diagnose
name = "SPINE"
requires_input_keys = ["prompt"]
cost_seconds_per_query = 12
interpret_class=util.constant.feature
external_internal = util.constant.internal_str
path = [external_internal, util.constant.representation, interpret_class]