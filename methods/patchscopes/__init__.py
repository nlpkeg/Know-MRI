from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class patchscopesHyperParams(HyperParams):
    lr_scale: float = None
    n_toks: int = None
    model_path: str = None
    refine: bool = None
    batch_size: int = None
    steps: int = None
    adaptive_threshold: float = None
    p: float = None
    need_mapping: bool = None

from .patchscopes_main import diagnose
name = "Patchscopes"
requires_input_keys = ["prompt", "ground_truth"]
cost_seconds_per_query = 3
interpret_class=util.constant.hiddenstates
external_internal = util.constant.internal_str
path = [external_internal, util.constant.representation, interpret_class]

