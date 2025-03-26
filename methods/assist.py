from methods import method_name2sub_module
from dataset_process import name2dataset_module
from pathlib import Path

def get_methods_by_dataset(dataset_name:str):
    method_names = []
    for method_name, method_module in method_name2sub_module.items():
        support_template_keys = name2dataset_module[dataset_name].support_template_keys
        requires_input_keys = method_module.requires_input_keys
        if set(support_template_keys).issuperset(requires_input_keys):
            method_names.append(method_name)
    return method_names

def get_methods_by_model_name(model_name:str):
    method_names = []
    for method_name, method_module in method_name2sub_module.items():
        hparam_path_of_method_model = Path(method_module.__file__).parent/"hparams"/(model_name.replace(r"/", "_")+".json")
        if hparam_path_of_method_model.exists():
            method_names.append(method_name)
    return method_names

def get_methods_by_dataset_and_model_name(dataset_name:str, model_name:str):
    methods1 = get_methods_by_dataset(dataset_name=dataset_name)
    methods2 = get_methods_by_model_name(model_name=model_name)
    return list(set(methods1).intersection(methods2))

def sort_methods_by_cost_time(method_names: list):
    return sorted(method_names, key=lambda name:method_name2sub_module[name].cost_seconds_per_query)