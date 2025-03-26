from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class SEHyperParams(HyperParams):
    model_path: str = None
    do_sample: bool = None
    max_length: int = None
    top_k: int = None
    top_p: float = None
    temperature: float = None
    repetition_penalty: float = None
    topk_words: int = None

   
from .SE_main import diagnose
name = "Self-explanation"
requires_input_keys = ["prompt", "ground_truth"]
cost_seconds_per_query = 3
interpret_class=util.constant.self_explanation
external_internal = util.constant.external_str
path = [external_internal, interpret_class]