import os
import sys
from pathlib import Path
base_dir = str(Path(__file__).absolute().parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)
import threading
import methods
from util.generate import get_model_output_

def get_model_output(sample, model_name_or_path, method=None, hparams=None):
    try:
        model_output = sample["ground_truth"]
    except:
        model_output = ""
    return model_output

def diagnosing(sample, model_name_or_path, method, hparams=None):
    # result = dict()
    diagnose_proxy = methods.method_name2diagnose_fun[method]
    result = diagnose_proxy(sample=sample, model_name_or_path=model_name_or_path, hparams=hparams)
    return result


if __name__ == "__main__":
    pass