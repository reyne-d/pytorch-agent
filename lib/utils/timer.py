import time
import torch
import numpy as np


def getSystemTime(format='%Y-%m-%d_%H:%M'):
    return time.strftime(format)


class Timer(object):
    def __init__(self, name='', every_n_step=1, with_cuda=False):
        self._with_cuda = with_cuda
        self._name = name
        self._count = 0
        self._start = 0.0
        self._end = 0.0
        self._every_n_step = every_n_step
        self._all_time = []

    def tic(self):
        if self._with_cuda:
            torch.cuda.synchronize()
        self._start = time.time()

    def toc(self, n=1):
        if self._with_cuda:
            torch.cuda.synchronize()
        self._end = time.time()
        self._count += n
        self._all_time.append(self._end - self._start)
        self._start = 0.0

    def average_time(self):
        if self._count == 0:
            return 0
        return np.sum(self._all_time) / float(self._count)

    def format_average_time(self, mile='ms'):
        if mile=='ms':
            fstr = '{} : {} ms'.format(self._name, self.average_time()*1000)
        elif mile == 's':
            fstr = '{} : {} s'.format(self._name, self.average_time())
        else:
            raise ValueError("Currently only support us or s")
        return fstr
