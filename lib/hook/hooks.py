"""
    This file pre-defined some common hooks, every hook has follow the same rules
    and will be called with the hook_runner.
    
    Predefine hooks : StopAtStepHook, LoggingPrintHook, TensorboardHook
"""


from lib.utils.timer import Timer
import torch
import numpy as np
import os
import pickle

from lib.agent.data_collector import CollectionKey
from lib.hook.hook_runner import _Hook
from lib.agent.model_obj import ModeKey
from lib.utils.envs import make_dir

import logging
logger = logging.getLogger()


class StopAtStepHook(_Hook):
    """
        Base counter for determining when `StopIteration` should trigger.
        start_iter: The current iteration.
        iters: run ${iters} times or None if provide max_iter
        max_iters: last iteration can achieve or None if provide iters
    """
    def __init__(self, start_iter, iters=None, max_iters=None):
        assert iters is not None or max_iters is not None, "must provide iters or max_iters."
        assert iters is None or max_iters is None, "can not provide iters and max_iters at the same time"
        self._max_iters = max_iters if max_iters is not None else start_iter + iters
        self._cur_iter = start_iter
        self._start_iter = start_iter
        super(StopAtStepHook, self).__init__()

    def should_stop(self):
        if self._cur_iter >= self._max_iters:
            raise StopIteration
        self._cur_iter += 1

    def before_run(self, data_collector):
        self.should_stop()

    def __repr__(self):
        return "StopAtStepHook(start_iter={}, max_iters={})".format(self._start_iter, self._max_iters)


class LoggingPrintHook(_Hook):
    def __init__(self, n_log_iter=None, log_at_end=False):
        """
            Prints the given values every N times, the values are provided by agent log_buff.
        :param n_log_iter: every N times to log
        """
        assert n_log_iter is not None or log_at_end

        self._n_log_iter = n_log_iter
        self._log_at_ent = log_at_end
        self.reset()
        super(LoggingPrintHook, self).__init__()

    def reset(self):
        self._iter_count_timer = 0
        self._cur_iter = 0

    def begin(self, data_collector):
        with_cuda = torch.cuda.device_count() > 0
        self._timer = Timer(with_cuda=with_cuda)
        self._timer.tic()

    def default_format_tensor(self, log_var, keys=None):
        # global_step = int(tensor_values[CollectionKey.GLOBAL_STEP].item())
        string = []
        torch.set_printoptions(precision=4)
        ori_option = np.get_printoptions()
        np.set_printoptions(precision=4)
        if keys is None:
            keys = log_var.keys()
        for name in keys:
            tensor = log_var[name]
            if isinstance(tensor, dict):
                tmp_str = name + " : " + self.default_format_tensor(log_var)
            else:
                tmp_str = "{}={:.4f}".format(name, tensor)
            string.append(tmp_str)
        np.set_printoptions(ori_option)
        return ', '.join(string)

    def __repr__(self):
        return "LoggingPrintHook(n_log_iter={})".format(self._n_log_iter)

    def _log_tensor(self, cur_iter, data_collector, is_training=True):

        torch.set_printoptions(profile='profile')

        log_buff = data_collector.get_var("log_buff")
        max_iter = data_collector.get_var("max_iters")

        if is_training:
            batch_time = self._timer.average_time()
            lr = data_collector.get_var("LR")
            log_buff.average(n=1)
            logger.info("Training [{}/{}] LR={:.8f}, batch_time={:.4f} iter/sec, {}".format(
                        cur_iter, max_iter, lr, batch_time,
                        self.default_format_tensor(log_buff.output)))
        else:
            log_buff.average()
            logger.info("Evaluate [{}/{}]: {}".format(
                cur_iter, max_iter, self.default_format_tensor(log_buff.output)))

    def after_run(self, data_collector):
        is_training = data_collector.get_var("mode") == ModeKey.TRAIN
        self._cur_iter += 1
        cur_iter = data_collector.get_var("global_step") if is_training else self._cur_iter

        if self._n_log_iter is not None and cur_iter % self._n_log_iter == 0:
            self._timer.toc(n=self._iter_count_timer)
            self._log_tensor(cur_iter, data_collector, is_training=is_training)
            self._iter_count_timer = 0
            self._timer.tic()
        self._iter_count_timer += 1

    def end(self, data_collector):
        is_training = data_collector.get_var("mode") == ModeKey.TRAIN
        if self._log_at_ent:
            cur_iter = data_collector.get_var("global_step")
            self._timer.toc(self._iter_count_timer)
            self._log_tensor(cur_iter, data_collector, is_training)

        self.reset()



class TensorboardHook(_Hook):
    """
        Log the given values every N times in tensorboard, the values are provided by agent log_buff.
        :param n_log_iter: every N times to log
    """
    def __init__(self, n_summary_iter=None):
        self._n_summary_iter = n_summary_iter
        super(TensorboardHook, self).__init__()

    def begin(self, data_collector):
        self._visualizer = data_collector.get_var(CollectionKey.VISUALIZER)

    def _add_to_summary(self, values, global_step):
        for key, val in values.items():
            self._visualizer.add_scalar(key, val, global_step=global_step)

    def _visualize_summary(self, data_collector, is_training):
        log_buff = data_collector.get_var("log_buff")
        cur_iter = data_collector.get_var("global_step")

        if is_training:
            log_buff.average(n=1)
            log_var = {"LR": data_collector.get_var("LR")}
            log_var.update(log_buff.output)
            tag = 'Training'
        else:
            log_buff.average()
            tag = 'Eval'
            log_var = log_buff.output
        for k, v in log_var.items():
            self._visualizer.add_scalar("%s.%s"%(tag, k), v, global_step=cur_iter)

    def after_run(self, data_collector):
        cur_iter = data_collector.get_var(CollectionKey.GLOBAL_STEP)
        is_training = data_collector.get_var("mode") == ModeKey.TRAIN
        if is_training and cur_iter > 0 and cur_iter % self._n_summary_iter == 0:
            self._visualize_summary(data_collector, is_training)

    def end(self, data_collector):
        is_training = data_collector.get_var("mode") == ModeKey.TRAIN
        if not is_training:
            self._visualize_summary(data_collector, is_training)


class CheckpointSaveHook(_Hook):
    """
        Save checkpoint every N times.
        n_save_period_iter: every N time to save
        save_at_end: whether save in the end of training
    """
    def __init__(self, n_save_period_iter=1000, save_at_end=True):
        assert n_save_period_iter > 0, "n_save_period_iter must greater than zero."
        self._n_save_period_iter = n_save_period_iter
        self._save_at_end = save_at_end
        super(CheckpointSaveHook, self).__init__()

    def _save(self, data_collector):
        model_obj = data_collector.get_var(CollectionKey.MODEL_OBJECT)
        cur_iter = data_collector.get_var(CollectionKey.GLOBAL_STEP)

        save_dir = data_collector.get_var("output_dir")
        save_path = os.path.join(save_dir, 'ckpt-{}'.format(cur_iter))

        model_obj.export(save_path, global_step=cur_iter)
        logger.info("Checkpoint is saved to {}".format(save_path))

    def after_run(self, data_collector):
        cur_iter = data_collector.get_var(CollectionKey.GLOBAL_STEP)
        if cur_iter > 0 and cur_iter % self._n_save_period_iter == 0:
            self._save(data_collector)

    def end(self, data_collector):
        cur_iter = data_collector.get_var(CollectionKey.GLOBAL_STEP)
        if self._save_at_end and cur_iter % self._n_save_period_iter != 0:
            self._save(data_collector)


class MeterLoggerHook(_Hook):
    """
       Tracking and save the data across the entire dataset and running globally processing.
       n_sample: the number of sample in a batch
       global_fn: it will be called in the end of iterations if not None.

    """
    def __init__(self, n_sample, global_fn=None):
        self.n_sample = n_sample
        super(MeterLoggerHook, self).__init__()
        self.global_fn = global_fn
        self.reset()

    def reset(self):
        self.datas = {}
        self._index = 0

    def after_run(self, data_collector):
        output = data_collector.get_var("run_output")
        var_to_record = output['var_to_record']
        batch_index = output.get('batch_index', None)
        if batch_index is None:
            n_sample = list(var_to_record.values())[0].shape[0]
            batch_index = np.arange(self._index, self._index+n_sample)
            self._index += n_sample

        for k, v in var_to_record.items():
            if k not in self.datas:
                if v.ndim==1:
                    self.datas[k] = np.zeros(self.n_sample)
                else:
                    shape = v.shape[1:]
                    self.datas[k] = np.zeros((self.n_sample, *shape))
            self.datas[k][batch_index] = v

    def end(self, data_collector):

        # save result to work_dir
        output_dir = data_collector.get_var("output_dir")
        mode = data_collector.get_var("mode")
        save_dir = os.path.join(output_dir, mode)
        cur_iter = data_collector.get_var("global_step")
        make_dir(save_dir)
        save_path = os.path.join(save_dir, '{}_step_{}_result.pkl'.format(mode, cur_iter))

        if self.global_fn is not None:
            for k, fun in self.global_fn.items():
                res = fun(self.datas)
                self.datas[k+'.result'] = res
        with open(save_path, 'wb') as f:
            pickle.dump(self.datas, f)
        logger.info("Data is saved to {}".format(save_path))
        self.reset()

