import torch

import os
import torch.optim.lr_scheduler
from lib.optims import build_optimizer, build_lr_schedule

import logging
logger = logging.getLogger("__main__.warm_up_lib")


class ModeKey(object):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'


class _ModelObj(object):
    """ A basic class for packaging model, optimizer, and lr_schedule."""
    def __init__(self, model, optimizer_cfg, lr_schedule_cfg):
        optimizer = build_optimizer(model.parameters(), optimizer_cfg)
        lr_schedule = build_lr_schedule(optimizer, lr_schedule_cfg)
        self._model = model
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule

    def register_forward_and_backward_hooks(self,args, **kwargs):
        """ A interface used to register hooks in model """
        pass

    def remove_forward_and_backward_hooks(self, args, **kwargs):
        """ A interface used to remove hooks in model """
        pass

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def predict(self):
        self._model.eval()

    def to(self, device):
        self._model.to(device)
        logger.info("Using device {}".format(device))

    def to_nn_DataParallel(self):
        if not isinstance(self._model, torch.nn.DataParallel):
            self._model = torch.nn.DataParallel(self._model)

    def export(self, save_path, model_only=False, global_step=None, is_best=False):
        if model_only:
            checkpoint = {'model': self._model.state_dict()}
        else:
            assert self._optimizer is not None and self._lr_schedule is not None and global_step is not None
            checkpoint = {
                'model': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'lr_schedule': self._lr_schedule.state_dict(),
                'global_step': global_step
            }
        torch.save(checkpoint, save_path)
        if is_best:
            dir_name = os.path.basename(save_path)
            file_name = 'model.best.ckpt'
            torch.split(checkpoint, os.path.join(dir_name, file_name))

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_schedule(self):
        return self._lr_schedule


class ModelObj(_ModelObj):
    """"""
    def __init__(self, model, optimizer_cfg, lr_schedule_cfg):
        super(ModelObj, self).__init__(model, optimizer_cfg, lr_schedule_cfg)



