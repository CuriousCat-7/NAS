import os
import sys
sys.path.insert(0, os.getcwd())
import unittest
from model import *
import torch
from torch import nn


class Test(unittest.TestCase):

    def test_metakernel(self):
        x = torch.rand(4, 3, 64, 64)
        conv = Conv2dMetaKernel(3, 4, [1,3,5], stride=1)
        y = conv(x)
        loss = y.mean()
        loss.backward()
        print("backward passed")

        conv.cuda()
        x = x.cuda()
        print("cuda", conv(x).shape)

        conv.eval()
        print(conv(x).shape)

        from utils import OnnxConverter
        OnnxConverter.convert_onnx(conv, "save/conv_metakernel.onnx",input_size=(1,3, 64,64))

    def test_ednetv2(self):
        model = ednetv2(num_classes=10)
        x = torch.rand(4, 3, 64, 64)
        y = model(x)
        print("cpu" ,y.shape)

        x = x.cuda()
        model.cuda()
        y = model(x)
        print("gpu" ,y.shape)
        y.sum().backward()

        model.set_temperature(1.0)
        for m in model.modules():
            if isinstance(m, Conv2dMetaKernel):
                self.assertEqual(m.temperature, 1.0)

        from utils import OnnxConverter
        model.eval()
        OnnxConverter.convert_onnx(model, "save/ednet_v2.onnx",input_size=(1,3, 64,64))



if __name__ == "__main__":
    unittest.main()
