from dataclasses import dataclass
from util.hparams import HyperParams
import util.constant

@dataclass
class EPHyperParams(HyperParams):
    model_path: str = None
    top_k: int = None
    token_nums: int = None
    dim: int = None
    method: str = None
    num_neighbors: int = None
    show_original_points: int = None
    show_similar_points: int = None
    show_original_tokens: int = None
    show_similar_tokens: int = None
    show_connection_lines: int = None
   
from .EP_main import diagnose
name = "EmbeddingProjection"
requires_input_keys = ["prompt"]
cost_seconds_per_query = 12
interpret_class=util.constant.embedding
external_internal = util.constant.internal_str
path = [external_internal, util.constant.module, interpret_class]