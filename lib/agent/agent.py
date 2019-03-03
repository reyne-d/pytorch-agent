import torch
import logging

from lib.hook import hooks
from lib.hook.hook_runner import HookRunner

from lib.agent.model_obj import ModeKey
from lib.agent.data_collector import CollectionKey, DataCollector
from lib.utils import make_dir
from lib.utils.timer import getSystemTime

from torch.utils.data import DataLoader
import os
import yacs
import copy
from lib.utils.log_buffer import LogBuffer


class Agent(object):
    """
        A agent for training, evaluating, predicting, it maintains all variables and data throughout the lifttime.
        Apart from some common components, such as model, optimizer and lr_schedule, the agent also includes two
        other important components, a hook_runner `agent.hook_runner` and a global data access`agent.data_collector`.
        The former comes with a code injection mechanism and the latter provides all the data the former needs.

        The training process will be running in iter-by-iter with hook.

        A simplified abstraction training process is as following:

        ```
        add_context_into_data_collector(data_collector)

        hook_runner.call_hooks_begin(data_collector)
        while cur_iter < max_iter:
            hook_runner.call_hooks_before_run(data_collector)

            output = run_a_iteration(inputs)
            add_output_to_data_collector()

            hook_runner.after_hooks_before_run(data_collector)

        hook_runner.call_hooks_before_end(data_collector)

        ```
    """
    def __init__(self, model_obj, config, logger, use_tensorboard=True):
        assert isinstance(config, yacs.config.CfgNode)
        config = copy.deepcopy(config)
        # setup config
        logger.info("Using config:\n{}".format(config))
        self._tag = getSystemTime()

        self._output_dir = config.path.output_dir
        make_dir(self._output_dir)

        save_path = self.save_config_to_output_dir(config)
        logger.info("Copy config to {}".format(save_path))

        if use_tensorboard:
            from lib.utils.tensorboardX_visualizer import tensorboardVisualizer
            summary_dir = os.path.join(self._output_dir, "tf_logs")
            make_dir(summary_dir)
            self._visualizer = tensorboardVisualizer(summary_dir)

        device = "cuda" if torch.cuda.device_count()>0 else "cpu"
        self._device = torch.device(device)
        self._model_obj = model_obj
        self._model_obj.to_nn_DataParallel()
        self._model_obj.to(self.device)

        self._data_collector = DataCollector()
        self._global_step = 0

        if use_tensorboard:
            self.update_collection_var(CollectionKey.VISUALIZER, self._visualizer)
        self.update_collection_var(CollectionKey.MODEL_OBJECT, model_obj)
        self.update_collection_var(CollectionKey.GLOBAL_STEP, self._global_step)
        self.update_collection_var("output_dir", self._output_dir)

        self.training_hooks = []
        self.eval_hooks = []
        self.predict_hooks = []
        self.config = config
        self.logger = logger

    def increase_global_step(self):
        self._global_step += 1
        self.update_collection_var(CollectionKey.GLOBAL_STEP, self._global_step)
        self.update_collection_var("LR", self.get_lr())

    def update_collection_list(self, name, value):
        self._data_collector.update_list(name, value)

    def update_collection_var(self, name, value):
        self._data_collector.update_var(name, value)

    def save_config_to_output_dir(self, config):
        cfg_str = config.dump()
        config_dir = os.path.join(self._output_dir, 'configs')
        make_dir(config_dir)

        save_path = os.path.join(config_dir, 'config-{}.yaml'.format(self._tag))
        with open(save_path, 'w') as f:
            f.write(cfg_str)
        return save_path

    def get_lr(self):
        return self._model_obj.lr_schedule.get_lr()[0]

    @property
    def data_collector(self):
        return self._data_collector

    @property
    def device(self):
        return self._device

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def global_step(self):
        return self._global_step

    def build_hook(self, hook_config, hook_op):
        return hook_op(**hook_config)

    def register_training_hook(self, log_config=None,
                               summary_config=None,
                               checkpoint_config=None):
        training_hooks = []
        if log_config is not None:
            training_hooks.append(self.build_hook(log_config, hooks.LoggingPrintHook))
        if summary_config is not None:
            training_hooks.append(self.build_hook(summary_config, hooks.TensorboardHook))
        if checkpoint_config is not None:
            training_hooks.append(self.build_hook(checkpoint_config, hooks.CheckpointSaveHook))

        self.training_hooks.extend(training_hooks)

    def register_hook(self, hook, mode):
        if mode == ModeKey.TRAIN:
            self.training_hooks.append(hook)
        elif mode == ModeKey.EVAL:
            self.eval_hooks.append(hook)
        elif mode == ModeKey.PREDICT:
            self.predict_hooks.append(hook)
        else:
            raise ValueError

    def register_evaluate_hook(self):
        self.eval_hooks.append(hooks.LoggingPrintHook(log_at_end=True))
        self.eval_hooks.append(hooks.TensorboardHook())

    def resume_model(self, ckpt_path, resume_optimizer=True, strict=False):
        checkpoint = torch.load(ckpt_path)
        model = self._model_obj.model
        model.load_state_dict(checkpoint['model'], strict=strict)
        self._global_step = checkpoint['global_step']
        if resume_optimizer:
            self._model_obj.optimizer.load_state_dict(checkpoint['optimizer'])
            self._model_obj.lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        logging.info("Resumed from {}".format(ckpt_path))
        self.logger.info('Resumed iter %d'%self._global_step)

    def load_pretrained(self, ckpt_path, strict=False):
        checkpoint = torch.load(ckpt_path)
        model = self._model_obj.model
        model.load_state_dict(checkpoint['model'], strict=strict)
        self.logger.info("Load checkpoint from {}".format(ckpt_path))

    def run(self, dataloaders, workflows, max_iters, model_fn,**kwargs):
        """
            Usage: agent.run([train_loader, eval_loader],
                             [('train', 1000),('eval', None),
                             max_iter=20000, )
        :param dataloaders:
        :param workflows:
        :param max_iters:
        :param other_param:
        :return:
        """
        assert max_iters is None or max_iters>0
        assert len(dataloaders) == len(workflows), "number of input dataloader must equal to workflows"
        self.update_collection_var('max_iters', max_iters)

        st = ["{} set: {}".format(workflows[i][0], len(dataloaders[i])) for i in range(len(workflows))]
        self.logger.info(", ".join(st))

        stop_at_one_workflow = max_iters is None
        max_iters = float("inf")
        while self._global_step < max_iters:
            for i, (mode, iters) in enumerate(workflows):
                data_iterator = make_data_iterator(dataloaders[i])
                batch_hook = list()
                if iters is not None:
                    batch_hook.append(hooks.StopAtStepHook(start_iter=self._global_step,
                                                           iters=iters))

                if mode == ModeKey.TRAIN:
                    batch_hook.extend(self.training_hooks)
                    self._model_obj.train()

                elif mode == ModeKey.EVAL:
                    batch_hook.extend(self.eval_hooks)
                    self._model_obj.eval()

                elif mode == ModeKey.PREDICT:
                    batch_hook.extend(self.predict_hooks)
                    self._model_obj.eval()
                else:
                    raise ValueError
                hook_runner = HookRunner(hooks=batch_hook)

                hook_runner.call_hooks_begin(self.data_collector)
                log_buff = LogBuffer()
                self.update_collection_var("log_buff", log_buff)
                self.update_collection_var("mode", mode)
                self.logger.info("running mode : {}".format(mode))

                while True:
                    try:
                        hook_runner.call_hooks_before_run(self.data_collector)

                        if mode == ModeKey.TRAIN:
                            features, labels = next(data_iterator)
                            run_output = model_fn(features, labels, mode, self._model_obj, **kwargs)
                            self.increase_global_step()

                        else:
                            with torch.no_grad():
                                features, labels = next(data_iterator)
                                run_output = model_fn(features, labels, mode, self._model_obj, **kwargs)
                        if "log_var" in run_output:
                            log_buff.update(run_output['log_var'], run_output['n_sample'])
                        self.update_collection_var("run_output", run_output)
                        hook_runner.call_hooks_after_run(self.data_collector)

                    except StopIteration:
                        break
                outputs = hook_runner.call_hooks_end(self.data_collector)
                self.update_collection_list('hook_ouput_%s' % mode, outputs)
            if stop_at_one_workflow:
                break


def make_data_iterator(input):
    """
        A adaptor to convert input to data_iterator
    :param input: Currently only support `torch.utils.data.DataLoader`
    :return:
    """
    assert isinstance(input, DataLoader)
    data_iterator = iter(input)
    return data_iterator