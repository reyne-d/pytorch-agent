import os
import torch
import numpy as np
import random
import logging
import six
import os.path as osp
logger = logging.getLogger()


def set_random_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    logger.info("Using random_seed: {}".format(seed))


def set_gpus(gpu_ids):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        logger.info("Using GPUs: {}".format(gpu_ids))


def make_dir(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)
