import unicodedata
from typing import List, Optional
from util.model_tokenizer import get_cached_model_tok
from transformers import AutoModelForCausalLM, AutoTokenizer
try:from transformers import GenerationConfig
except:pass

def get_model_output_(
    sample,
    model_name_or_path,
    method=None,
    hparams=None,
    top_k: int = 40,
    max_out_len: int = 200,
):
    return sample["ground_truth"]
