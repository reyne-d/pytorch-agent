import torch
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLRSchedule(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0/3,
                 warmup_iter=500, warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers, got {}".format(milestones))

        if warmup_method not in ['constant', 'linear']:
            raise ValueError("Only 'constant' or 'linear' accepted, got {}".format(warmup_method))

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iter = warmup_iter
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        super(WarmupMultiStepLRSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warm_factor = 1
        if self.last_epoch < self.warmup_iter:
            if self.warmup_method == 'constant':
                warm_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iter
                warm_factor = self.warmup_factor * (1-alpha) + alpha
        lrs = [
            base_lr * warm_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

        return lrs

