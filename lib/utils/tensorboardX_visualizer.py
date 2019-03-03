import tensorboardX as tfX
import os
import torch
import numpy as np
import torchvision.utils as vutils
from lib.utils.timer import getSystemTime
from lib.utils.dict_op import slice_dic, flatten_dic
from lib.utils.image_process import add_text_in_img
import cv2

class tensorboardVisualizer(tfX.SummaryWriter):
    """
            A subclass of tensorboard SummaryWriter

    """
    def __init__(self, log_dir='logs', comment='',format='%(asctime)s-%(comment)s'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        ti = getSystemTime()
        path = format % {'asctime': ti, 'comment': comment}
        super(tensorboardVisualizer, self).__init__(log_dir=log_dir, comment=path)

    @classmethod
    def print_confusion_matrix_with_markdown(self, mat, labels=None):
        n_class = len(mat)
        if isinstance(labels, dict):
            assert (len(labels) == n_class)
            id_2_label = labels
        if isinstance(labels, list):
            assert (len(labels) == n_class)
            id_2_label = {i: labels[i] for i in range(n_class)}
        if labels is None:
            id_2_label = {i: i for i in range(n_class)}
        assert isinstance(id_2_label, dict),"Invalid labels, expect dict,list or None, but got {}".format(labels)
        su = float(np.sum(mat))
        fstr = ''' Actural/Prediction ''' + ''.join(['''| {} '''.format(id_2_label[i]) for i in range(n_class)]) + '''\n'''
        fstr += ''.join([' - |' for i in range(n_class)]) + '''-\n'''
        for i, row in enumerate(mat):
            fstr += ' **{}** '.format(id_2_label[i]) + \
                    ''.join(['| {} ({:.2f}%) '.format(c, c * 100 / su) for c in row]) + '''\n'''
        return fstr

    def plot_image_batch(self, imgs, name, n_iter=0, n_row=8, n_example=-1, norm_param=None):
        n_example = np.clip(n_example, -1, imgs.size(0))
        if norm_param is not None:
            imgs = imgs.permute(0, 2, 3, 1)  # NCHW -> NHWC
            imgs *= torch.Tensor(norm_param['std'])
            imgs += torch.Tensor(norm_param['mean'])
            imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x = vutils.make_grid(imgs[:n_example], nrow=n_row, normalize=False, scale_each=False)
        self.add_image(name, x, n_iter)

    def plot_graph(self, model, input_size):
        dummy_input = torch.zeros(1,3, input_size[0], input_size[1]).cuda()
        self.add_graph(model, dummy_input, True)

    def print_format_metrics(self, tag, confusion_mat=None, labels=None, global_step=None, walltime=None, **kwargs):
        fstr = ''
        if confusion_mat is not None:
            fstr += self.print_confusion_matrix_with_markdown(confusion_mat, labels)

        for k, v in kwargs.items():
            fstr += '''**{}** : {:.4f}\n'''.format(k, v)
        self.add_text(tag, fstr,
                      global_step=global_step, walltime=walltime)

    def plot_confusion_matrix(self, tag, confusion_mat, labels=None, global_step=None, walltime=None):
        fstr = '''Confusion Matrix.0` | ''' + \
               self.print_confusion_matrix_with_markdown(confusion_mat, labels=labels)
        self.add_text(tag, fstr, global_step=global_step, walltime=walltime)

    def plot_model_parameters(self, named_parameter, global_step=None, walltime=None):
        for name, param in named_parameter:
            if 'bn' not in name:
                self.add_histogram(name, param, global_step=global_step, walltime=walltime)

    def plot_embedding(self,mat, metadata=None, label_img=None, global_step=None,
                       tag='default', metadata_header=None,norm_param=None):
        if norm_param is not None:
            imgs = label_img.permute(0, 2, 3, 1)  # NCHW -> NHWC
            imgs *= torch.Tensor(norm_param['std'])
            imgs += torch.Tensor(norm_param['mean'])
            label_img = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        self.add_embedding(mat,metadata=metadata,label_img=label_img,
                           tag=tag, global_step=global_step, metadata_header=metadata_header)

    def plot_epoch_result(self, headers, global_step=None,walltime=None,**kwargs):
        assert isinstance(headers, (list, set)), "list or set are expected"
        assert all([isinstance(v, (list, set)) for v in kwargs.values()]), "list or set are expected"
        assert all([len(v) == len(headers) for v in kwargs.values()])
        for k, v in kwargs.items():
            self.add_scalars('Summary_%s'%k, dict(zip(headers, v)),
                                 global_step=global_step, walltime=walltime)

    def visual_image_with_text(self,tag, imgs, print_dic,n_row,save_dir, norm_param=None,
                               is_scale=True, global_step=None, walltime=None):
        assert all(len(imgs) == len(data) for data in print_dic.values()), \
            "imgs and print_dic should have same length"

        if norm_param is not None:
            imgs = imgs.permute(0, 2, 3, 1)  # NCHW -> NHWC
            imgs *= torch.Tensor(norm_param['std'])
            imgs += torch.Tensor(norm_param['mean'])
            imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        text_imgs = add_text_in_img(imgs.numpy(), print_dic,is_scale=is_scale)
        text_imgs = torch.Tensor(text_imgs)

        x = vutils.make_grid(text_imgs, nrow=n_row)
        save_path = os.path.join(save_dir, '%s_%d.png'%(tag, global_step))
        cv2.imwrite(save_path, (x.numpy().transpose(1,2,0)[:,:,(2,1,0)])*255)
        self.add_image(tag, x, global_step=global_step,walltime=walltime)





