# Introduction
Pytorch Agent is a high-level API that simplifies deep learning programming. It is inspired by [Tensorflow.Estimator](https://github.com/tensorflow/estimator) and is a simplified pytorch-based version.

Agent encapsulated training, evaluation and prediction to support implement new algorithms quickly. It implements hook mechanism to provide secure code injection and provide predefined hooks.

# Usage

Train a model: 
```shell
    python train_cifar_resnet.py cifar10_config.yaml 0,1
``` 

| Net      | Accuracy |
| -------- | -------- |
| [Resnet18](https://arxiv.org/abs/1512.03385)  | 0.9536 |

Test a model:
```shell
    python test_cifar_resnet.py cifar10_config.yaml 0,1 solver.load_from checkpoints/cifar_resnet.ckpt
```
