from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class CausalTraceHyperParams(HyperParams):
    model_path: str = None
    replace: bool = None
    window: int = 10
    noise_level: str = None # s, m, t, u

from .causal_trace_main import diagnose
name = "CausalTracing"
requires_input_keys = ["prompt", "triple_subject"]
cost_seconds_per_query = 180
interpret_class=util.constant.mlp
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, interpret_class]