from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class AttributionHyperParams(HyperParams):
    model_path: str = None
    num_steps: int = None
    batch_size: int = None

from .attribution_main import diagnose
name = "Attribution"
requires_input_keys = ["prompt", "ground_truth"]
cost_seconds_per_query = 3
interpret_class=util.constant.Attribution
external_internal = util.constant.external_str
path = [external_internal, interpret_class]
