import json
from dataclasses import dataclass
import inspect
from pathlib import Path

@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_model_name_or_path(cls, model_name_or_path):
        model_name_or_path = model_name_or_path.replace(r"/", "_")
        hparams_json_path = Path(inspect.getfile(cls)).parent/"hparams"/(model_name_or_path+".json")
        return cls.from_json(hparams_json_path)
    
