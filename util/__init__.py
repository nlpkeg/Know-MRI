import os
import sys
from pathlib import Path
base_dir = str(Path(__file__).absolute().parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)
os.environ["HF_HOME"] = "/netcache/huggingface"