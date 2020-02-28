
class Conv2dMetaKernel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size:list,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 temperature_start=5.0):
        super(Conv2dMetaKernel, self).__init__()
        kernel_size.sort()
        self.assert_kernel(kernel_size)

        torch.nn.Conv2d
        self.weight_list = []
        self.bias_list = []
        self.range_hs = []
        self.range_ws = []
        max_k = kernel_size[-1]
        for k in kernel_size:
            weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, k, k))
            self.weight_list.append(weight)
            if bias:
                b = nn.Parameter(torch.Tensor(out_channels))
                self.bias_list.append(b)
            diff = (max_k - k)//2
            self.range_hs.append( (diff, max_k - diff) )
            self.range_ws.append( (diff, max_k - diff) )

        self.theta = nn.Parameter(torch.ones((len(kernel_size))))
        self.padding_list = [_pair(k//2) for k in kernel_size]

        self.stride = _pair(stride)
        self.padding = self.padding_list[-1]
        self.dilation = _pair(dilation)
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode
        self.temperature = temperature_start

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
                bias = False
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            arg_theta = self.arg_theta
            weight = self.weight_list[arg_theta]
            padding = self.padding_list[arg_theta]
            bias = self.bias_list[arg_theta]
            return F.conv2d(input, weight, bias, self.stride,
                            padding, self.dilation, self.groups)

    def eval(self):
        self.arg_theta = self.theta.max(0)[1]
        super(Conv2dMetaKernel, self).eval()


