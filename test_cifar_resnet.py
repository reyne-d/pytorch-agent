import logging
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
from torch.autograd import Variable
import argparse
from classify_config import cfg
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from lib.utils import set_gpus, set_random_seed, getSystemTime, print_forward_tensor, setup_logger
from lib.utils.metrics import accuracy
from lib.data.data_loader import build_data_loader
from lib.agent.agent import Agent
from lib.hook import hooks
from lib.agent.model_obj import ModelObj, ModeKey
import resnet_cifar

import os.path as osp


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, default=None,
                        help="A customer config file")
    parser.add_argument("gpu_ids", type=str, default='',
                        help="Args for experiment")
    parser.add_argument("opts", default=None,
                        nargs=argparse.REMAINDER,
                        help="Modify model config options using the command-line",)
    args = parser.parse_args()
    return args


def train_val_fn(features, labels, mode, model_obj):
    imgs = Variable(features.cuda())
    labels = Variable(labels.long().cuda())

    outputs = model_obj.model(imgs)
    loss = model_obj.loss_fn(outputs, labels)
    acc = accuracy(outputs, labels)

    log_var = dict(loss=loss.item(), acc=acc.item())
    _, pred = outputs.topk(1, 1, True, True)

    ret_dic = dict(log_var=log_var, n_sample=features.size(0))
    ret_dic['var_to_record'] = {'prediction': pred.data.cpu().numpy(),
                                'labels': labels.data.cpu().numpy()}
    return ret_dic


def global_accuracy(datas):
    y_pred = datas['prediction']
    y_true = datas['labels']
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    result = OrderedDict()
    result["acc#all"] = acc
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(len(classes)):
        index = y_true == i
        result[classes[i]] = accuracy_score(y_pred=y_pred[index], y_true=y_true[index])
    logger = logging.getLogger('__main__')
    logger.info(result)
    return result


if __name__ == '__main__':
    args = parser_args()
    cfg.merge_from_file(args.yaml)
    cfg.merge_from_list(args.opts)

    set_gpus(args.gpu_ids)
    set_random_seed(cfg.random_seed)
    cfg.freeze()

    logger = setup_logger(log_path=osp.join(cfg.path.output_dir, 'log-{}.txt').format(getSystemTime()),
                          level=logging.INFO)

    normalize = transforms.Normalize(mean=cfg.data.norm_param[0], std=cfg.data.norm_param[0])
    val_dataset = datasets.CIFAR10(root=cfg.path.data_dir, train=False, download=True,
                                   transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize]))

    n_gpu = len(args.gpu_ids.split(','))
    batch_size = cfg.data.batch_size_per_gpu * n_gpu
    num_workers = cfg.data.num_worker_per_gpu * n_gpu

    val_loader = build_data_loader(val_dataset, loop=False, shuffle=False,
                                   batch_size=batch_size, num_workers=num_workers)

    logger.info("eval dataset: {}".format(len(val_dataset)))

    # =============build model_obj ==============================================
    model_ft = getattr(resnet_cifar, cfg.model)()
    model_ft.fc = nn.Linear(2048, 10)
    model_obj = ModelObj(model_ft, optimizer_cfg=cfg.optimizer_cfg, lr_schedule_cfg=cfg.lr_schedule_cfg)
    model_obj.loss_fn = nn.CrossEntropyLoss()

    # =============build agent ==============================================
    agent = Agent(model_obj, cfg, logger, use_tensorboard=True)

    agent.register_evaluate_hook()
    agent.register_hook(hook=hooks.MeterLoggerHook(len(val_loader), global_fn={"accuracy": global_accuracy}),
                        mode=ModeKey.EVAL)

    agent.load_pretrained(cfg.solver.load_from)

    agent.run([val_loader],
              [('eval', None)],
              max_iters=None,
              model_fn=train_val_fn)
