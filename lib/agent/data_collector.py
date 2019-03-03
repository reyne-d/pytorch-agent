import logging
from collections import OrderedDict
logger = logging.getLogger('__main__.collection_manager')


class CollectionKey(object):
    MODEL = 'model'
    OPTIMIZER = 'optimizer'
    LR_SCHEDULE = 'lr_schedule'
    GLOBAL_STEP = 'global_step'
    LOSS = 'loss'
    LABELS = 'labels'
    OUTPUTS = 'outputs'
    PREDICTIONS = 'predictions'
    PROBABILITIES = 'probabilities'
    VAR_TO_LOG = 'var_to_log'
    EVAL_METRICS = 'eval_metrics'
    VISUALIZER = 'visualizer'
    FORWARD_HOOK_OUTPUT = 'forward_hook_output'
    BACKWARD_HOOK_OUTPUT = 'backward_hook_output'
    BATCH_SIZE = 'batch_size'
    MODEL_OBJECT = 'model_obj'


class DataCollector(object):
    """ A object used to access arbitrary data."""
    def __init__(self):
        """
            _collections: data container
            _finalize: if finalize, forbid to change collections, whether add or update.
        """
        self._collections_list = OrderedDict() # collection list
        self._collections_var = OrderedDict() # collect single object

    @property
    def lists(self):
        return self._collections_list

    def update_list(self, name, value):
        if name not in self._collections_list:
            self._collections_list[name] = [value]
        else:
            self._collections_list[name].append(value)

    def vars(self):
        return self._collections_var

    def update_var(self, name, value):
        self._collections_var[name] = value

    def get_var(self, name):
        return self._collections_var.get(name)

    def get_list(self, name):
        return self._collections_list.get(name)

