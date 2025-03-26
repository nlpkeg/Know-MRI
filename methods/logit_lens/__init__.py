from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class LogitLensHyperParams(HyperParams):
    model_path: str = None
    unembedding_num: int = None

from .logit_lens_main import diagnose
name = "Logit Lens"
requires_input_keys = ["prompt"]
cost_seconds_per_query = 1
interpret_class=util.constant.hiddenstates
external_internal = util.constant.internal_str
path = [external_internal, util.constant.representation, interpret_class]
