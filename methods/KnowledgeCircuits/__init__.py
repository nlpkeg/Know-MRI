from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class KnowledgeCircuitsHyperParams(HyperParams):
    model_path: str = None
    model_name: str = None
    api_key: str = None
    base_url: str = None
    topn: int = None
    

from .KnowledgeCircuits_main import diagnose
name = "KnowledgeCircuits"
requires_input_keys = ["prompt", "ground_truth", "triple_subject", "triple_relation", "triple_object"]
cost_seconds_per_query = 320
interpret_class=util.constant.circuit
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, interpret_class]





