import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import logging

#from utils import AvgrageMeter, weights_init, \
#                  CosineDecayLR
from data_parallel import DataParallel

class Conv2dMetaKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size:list,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(Conv2dMetaKernel, self).__init__()
        assert False
        kernel_size.sort()
        self.assert_kernel(kernel_size)

        self.weight_list = []
        if bias:
            self.bias_list = []
        self.range_hs = []
        self.range_ws = []
        self.max_k = kernel_size[-1]
        for k in kernel_size:
            weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, k, k))
            self.weight_list.append(weight)
            if bias:
                b = nn.Parameter(torch.Tensor(out_channels))
                self.bias_list.append(b)
            diff = (max_k - range_k)//2
            self.range_hs.append( (diff, max_k - diff) )
            self.range_ws.append( (diff, max_k - diff) )

        self.theta = nn.Parameter(torch.ones((len(kernel_size))))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

    def assert_kernel(self, kernel_size):
        assert isinstance(kernel_size, list)
        for k in kernel_size:
            assert isinstance(k, int)
            assert k % 2 == 1 # k must be 1, 3, 5, 7 ...
        assert len(set(kernel_size)) == len(kernel_size)


    def conv2d_forward(self ,x, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.training:
            assert False
            weight = torch.zeros_like(self.weight_list[-1])
            for idx in range(len(weight_list)):
                cur_weight = weight_list[idx] * thetas[idx]
                range_h = self.range_hs[idx]
                range_w = self.range_ws[idx]
                weight[:,
                       :,
                       range_h[0]:range_h[1],
                       range_w[0]:range_w[1]] += cur_weight
            return self.conv2d_forward(self, x, weight)
        else:
            pass


class EDNet(nn.Module):

    def __init__(self,):
        pass

    def forward(self):
        pass
