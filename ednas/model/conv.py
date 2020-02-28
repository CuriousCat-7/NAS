import torch
import torch.nn as nn
import torch.nn.functional as F
from data_parallel import DataParallel

_pair = torch.nn.modules.utils._pair


class Conv2dMetaKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size:list,
                 stride=1,
                 groups=1,
                 bias=True,
                 temperature_start=5.0):
        super(Conv2dMetaKernel, self).__init__()
        kernel_size.sort()
        self.assert_kernel(kernel_size)
        self.conv_list = nn.ModuleList()
        max_k = kernel_size[-1]
        self.range_hs = []
        self.range_ws = []

        for k in kernel_size:
            self.conv_list.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    k,
                    stride,
                    padding=k//2,
                    groups=groups,
                    bias=bias)
            )
            diff = (max_k - k)//2
            self.range_hs.append( (diff, max_k - diff) )
            self.range_ws.append( (diff, max_k - diff) )


        self.theta = nn.Parameter(torch.ones((len(kernel_size))))
        self.temperature = temperature_start
        self.weight_list = [conv.weight for conv in self.conv_list]
        self.bias_list = [conv.bias for conv in self.conv_list] if bias else []
        self.stride = self.conv_list[-1].stride
        self.padding = self.conv_list[-1].padding
        self.groups = self.conv_list[-1].groups
        self.dilation = self.conv_list[-1].dilation

    def assert_kernel(self, kernel_size):
        assert isinstance(kernel_size, list)
        for k in kernel_size:
            assert isinstance(k, int)
            assert k % 2 == 1 # k must be 1, 3, 5, 7 ...
        assert len(set(kernel_size)) == len(kernel_size)

    def forward(self, input):
        if self.training:
            theta = F.gumbel_softmax(self.theta, tau=self.temperature, hard=False)
            weight = torch.zeros_like(self.weight_list[-1])
            for idx in range(len(self.weight_list)):
                cur_weight = self.weight_list[idx] * theta[idx]
                range_h = self.range_hs[idx]
                range_w = self.range_ws[idx]
                weight[:,
                       :,
                       range_h[0]:range_h[1],
                       range_w[0]:range_w[1]] += cur_weight
            if self.bias_list:
                bias = torch.zeros_like(self.bias_list[-1])
                for idx in range(len(self.bias_list)):
                    bias += self.bias_list[idx] * theta[idx]
            else:
                bias = None
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return self.conv_list[self.arg_theta](input)

    def eval(self):
        self.arg_theta = self.theta.max(0)[1]
        super(Conv2dMetaKernel, self).eval()
