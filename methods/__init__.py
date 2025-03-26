import methods
import pkgutil
import importlib

method_name2diagnose_fun = dict()
method_name2sub_module = dict()
# from methods import kn
# methodname2module[kn.name] = kn.diagnose

for importer, modname, ispkg in pkgutil.iter_modules(methods.__path__):
    if ispkg:
        try:
            submodule = importlib.import_module(methods.__name__ + '.' + modname)
            method_name2diagnose_fun[submodule.name] = submodule.diagnose
            method_name2sub_module[submodule.name] = submodule
        except Exception as e:
            print(e)

support_methods = list(method_name2diagnose_fun.keys())