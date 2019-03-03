import logging
import torch.nn as nn
from torch.autograd import Variable
import argparse
from classify_config import cfg
from torchvision import datasets, transforms
import torch.nn.init as init
from lib.utils import set_gpus, set_random_seed, getSystemTime, print_forward_tensor, setup_logger
from lib.utils.metrics import accuracy
from lib.hook import hooks
from lib.data.data_loader import build_data_loader
from lib.agent.agent import Agent
from lib.agent.model_obj import ModelObj, ModeKey
import resnet_cifar
from sklearn.metrics import accuracy_score
from collections import  OrderedDict
import os.path as osp


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, default=None,
                        help="A customer config file")
    parser.add_argument("gpu_ids", type=str, default='',
                        help="Args for experiment")
    parser.add_argument("-opts", type=list, default=[],
                        help="Optional args")
    parser.add_argument("-debug", action="store_true", default=False,
                        help="whether run in debug mode")
    args = parser.parse_args()
    return args


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def train_val_fn(features, labels, mode, model_obj):
    imgs = Variable(features.cuda())
    labels = Variable(labels.long().cuda())

    outputs = model_obj.model(imgs)
    loss = model_obj.loss_fn(outputs, labels)
    acc = accuracy(outputs, labels)

    if mode == ModeKey.TRAIN:
        model_obj.optimizer.zero_grad()
        loss.backward()
        model_obj.optimizer.step()
        model_obj.lr_schedule.step()

    log_var = dict(loss=loss.item(), acc=acc.item())
    ret_dic = dict(log_var=log_var, n_sample=features.size(0))

    if mode == ModeKey.EVAL:
        _, pred = outputs.topk(1, 1, True, True)
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

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_path=osp.join(cfg.path.output_dir, 'log-{}.txt').format(getSystemTime()),
                          level=log_level)

    normalize = transforms.Normalize(mean=cfg.data.norm_param[0], std=cfg.data.norm_param[0])
    train_dataset = datasets.CIFAR10(root=cfg.path.data_dir, train=True, download=True,
                                     transform=transforms.Compose([
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize]))
    val_dataset = datasets.CIFAR10(root=cfg.path.data_dir, train=False, download=True,
                                   transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize]))

    n_gpu = len(args.gpu_ids.split(','))
    batch_size = cfg.data.batch_size_per_gpu * n_gpu
    num_workers = cfg.data.num_worker_per_gpu * n_gpu

    train_loader = build_data_loader(train_dataset, loop=True, shuffle=True,
                                     batch_size=batch_size, num_workers=num_workers)
    val_loader = build_data_loader(val_dataset, loop=False, shuffle=False,
                                   batch_size=batch_size, num_workers=num_workers)

    logger.info("train dataset: {}, eval dataset: {}".format(len(train_dataset), len(val_dataset)))

    # =============build model_obj ==============================================
    model_ft = getattr(resnet_cifar, cfg.model)()
    model_ft.fc = nn.Linear(2048, 10)
    init_params(model_ft)
    model_obj = ModelObj(model_ft, optimizer_cfg=cfg.optimizer_cfg, lr_schedule_cfg=cfg.lr_schedule_cfg)
    model_obj.loss_fn = nn.CrossEntropyLoss()

    if args.debug:
        print_forward_tensor(train_loader, net=model_obj.model)

    # =============build agent ==============================================
    agent = Agent(model_obj, cfg, logger, use_tensorboard=True)
    agent.register_training_hook(cfg.log_config, cfg.summary_config, cfg.checkpoint_config)
    agent.register_evaluate_hook()
    # agent.register_hook(hooks.LoggingPrintHook(n_log_iter=20), mode=ModeKey.EVAL)
    agent.register_hook(hook=hooks.MeterLoggerHook(len(val_loader), global_fn={"accuracy": global_accuracy}),
                        mode=ModeKey.EVAL)

    if cfg.solver.resume_from:
        agent.resume_model(cfg.solver.resume_from)
    elif cfg.solver.load_from:
        agent.load_pretrained(cfg.solver.load_from)

    agent.run([train_loader, val_loader],
              [('train', cfg.solver.eval_period), ('eval', None)],
              max_iters=cfg.solver.max_iters,
              model_fn=train_val_fn)

