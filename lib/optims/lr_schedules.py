from torch.optim.lr_scheduler import MultiStepLR, \
                        LambdaLR, StepLR, ExponentialLR
from .warmup_lr_schedule import WarmupMultiStepLRSchedule


__all__ = ['MultiStepLR', 'LambdaLR', 'StepLR', 'ExponentialLR', 'WarmupMultiStepLRSchedule']


def build_lr_schedule(optimizer, lr_schedule_cfg):
    """ Build lr_schedule from config."""
    lr_schedule_cfg = {key.lower(): value for key, value in lr_schedule_cfg.items()}
    schedule_type = lr_schedule_cfg['type']
    del lr_schedule_cfg['type']
    if schedule_type in __all__:
        return globals()[schedule_type](optimizer, **lr_schedule_cfg)
    raise Exception('The lr_schedule type {} does not exist in {}'.format(
        schedule_type, __all__
    ))
