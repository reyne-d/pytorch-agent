from yacs.config import CfgNode as CN
import os
from lib.utils.logger import setup_logger
import logging

logger = logging.getLogger('__main__.config')


_C = CN()

_C.agent_name = ''
_C.random_seed = 666
_C.model = ""
# ==============================================
# path SETTING
_C.path = CN()
_C.path.work_dir = os.getcwd()
# which use to save log and checkpoints
_C.path.output_dir = os.path.join(_C.path.work_dir, 'outputs', _C.agent_name)
_C.path.data_dir = os.path.join(_C.path.work_dir, 'data_dir')
# ==============================================


# ==============================================
# solver
_C.solver = CN()

_C.solver.max_iters= 20000
_C.solver.eval_period = 1000
_C.solver.resume_from = ''
_C.solver.load_from = ''
# ==============================================

# ==========================================
_C.log_config = CN()
_C.log_config.n_log_iter = 10
_C.summary_config = CN()
_C.summary_config.n_summary_iter = 10
_C.checkpoint_config = CN()
_C.checkpoint_config.n_save_period_iter = _C.solver.eval_period
_C.checkpoint_config.save_at_end = True
# ==========================================

# ==============================================
# optimizer_cfg SETTING
_C.optimizer_cfg = CN()
# The other filed expect type and params will be pass into pytorch optimizer_cfg.
# so than should be specify with the type of optimizer_cfg
_C.optimizer_cfg.type = 'SGD'
_C.optimizer_cfg.lr = 0.0005
_C.optimizer_cfg.weight_decay = 0.0005
_C.optimizer_cfg.momentum = 0.9
# _C.optimizer_cfg.WEIGHT_DECAY_BIAS = 0
# _C.optimizer_cfg.BIAS_LR_FACTOR = 2
# If not None, PARAMS must be a dict of dict, i.e. {'conv.0':  {'LR': 0.005, 'WEIGHT_DECAY':0.00001}
# which key represent the layer name, and value represent the hyperparams need to be overwritten
# _C.optimizer_cfg.PARAMS = None
# ==============================================

# ==============================================
# LR SETTING
_C.lr_schedule_cfg = CN()
_C.lr_schedule_cfg.type = 'WarmupMultiStepLRSchedule'
_C.lr_schedule_cfg.milestones = [10000, 15000]
_C.lr_schedule_cfg.gamma = 0.1
# ==============================================


# ==============================================
_C.data = CN()
_C.data.num_worker_per_gpu = 3

_C.data.batch_size_per_gpu = 32
_C.data.num_worker_per_gpu = 2
_C.data.norm_param = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # mean, std
# ==============================================

cfg = _C