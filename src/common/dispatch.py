from src.tasks.qed.task_clf.module import QedModule
from src.tasks.qed.task_adapt.module import QedAdaptModule


def modulename2cls(name):
    dict_globalname2obj = globals().copy()
    try:
        cls = dict_globalname2obj[name]
    except:
        raise NotImplementedError
    return cls