from _solution.tasks.test_task.module import MnistModule
from _solution.tasks.qed.task_clf.module import QedModule
from _solution.tasks.qed.task_adapt.module import QedAdaptModule
from _solution.tasks.flowers.module import FlowersModule
# from _solution.tasks.conditions.module import ConditionsModule

# TODO remove above

def modulename2cls(name):
    dict_globalname2obj = globals().copy()
    try:
        cls = dict_globalname2obj[name]
    except:
        raise NotImplementedError
    return cls