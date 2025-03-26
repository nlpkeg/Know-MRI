from pathlib import Path
import importlib.util
import sys

dataset_list = []
name2dataset_module = dict()

for module_file_path in Path(__file__).absolute().parent.glob("*.py"):
    if "__" in str(module_file_path):
        continue
    try:
        module_name = module_file_path.name.split(".")[0]  
        spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        name2dataset_module[module.dataset_info["name"]] = module
        if "user_input" not in module_file_path.name:
            dataset_list.append(module.dataset_info)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    print(name2dataset_module, "\n", dataset_list)