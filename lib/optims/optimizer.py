from torch.optim import SGD, Adam, Adadelta, RMSprop, Adagrad

__all__ = ['SGD', 'Adam', 'Adadelta',
           'RMSprop', 'Adagrad']


def build_optimizer(model_params, optimizer_cfg):
    """build optimizer from config"""
    optimizer_cfg = {key.lower(): value for key, value in optimizer_cfg.items()}

    optim_type = optimizer_cfg['type']
    del optimizer_cfg['type']
    if optim_type in __all__:
        return globals()[optim_type](model_params, **optimizer_cfg)
    else:
        raise Exception('The optimizer type {} does not exist in {}'.format(
            optim_type, __all__))





