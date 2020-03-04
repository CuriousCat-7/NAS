import torch
from loguru import logger


class Clamp_ReLU6(torch.nn.Module):

    def __init__(self, inplace=False):
        super(Clamp_ReLU6, self).__init__()
        self.inplace = inplace

    #@weak_script_method
    def forward(self, input):
        return input.clamp_(0, 6) if self.inplace else input.clamp(0,6)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class OnnxConverter():
    @classmethod
    def replace_ctc(cls, model, img_height):
        if hasattr(model, "_modules"):
            for name, m in model._modules.items():
                if isinstance(m, torch.nn.ReLU6):
                    model._modules[name] = Clamp_ReLU6(inplace=m.inplace)
                elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
                    # to replace nn.AdaptiveAvgPool2d((None, 1))
                    assert img_height % 16 == 0
                    model._modules[name] = torch.nn.AvgPool2d( kernel_size=(1, img_height//16), stride=(1, img_height//16) )
                else:
                    cls.replace_ctc(m, img_height)


    @classmethod
    def replace_pse(cls, model):
        if hasattr(model, "_modules"):
            for name, m in model._modules.items():
                if isinstance(m, torch.nn.ReLU6):
                    model._modules[name] = Clamp_ReLU6(inplace=m.inplace)
                else:
                    cls.replace_pse(m)

    @classmethod
    def replace_global_avg_pool(cls, model, kernel_size):
        if hasattr(model, "_modules"):
            for name, m in model._modules.items():
                if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                    model._modules[name] = torch.nn.AvgPool2d(kernel_size)
                else:
                    cls.replace_global_avg_pool(m, kernel_size)

    @classmethod
    def replace_relu6relu(cls, model):
        if hasattr(model, "_modules"):
            for name, m in model._modules.items():
                if isinstance(m, torch.nn.ReLU6):
                    logger.warning("replace relu6 to relu, this will make model act differently from original")
                    model._modules[name] = torch.nn.ReLU(inplace=m.inplace)
                else:
                    cls.replace_relu6relu(m)


    @staticmethod
    def convert_onnx(model, onnx_file="save/model.onnx", replace_fn=None, input_size=[1, 3, 640, 640], opset_version=10, verbose=False, checkonnx=True)->str:
        device = "cpu"
        model.to(device)
        dummy_input = torch.zeros(*input_size)
        if replace_fn is not None:
            replace_fn(model)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            logger.debug("model output rst in dummy_input is {}", output)
            torch.onnx.export(model,
                              dummy_input,
                              onnx_file,
                              verbose=verbose,
                              opset_version=opset_version)
        logger.success("onnx convert finish, save to {}", onnx_file)
        if not checkonnx:
            return onnx_file
        # valid onnx
        try:
            import onnx
        except:
            raise Exception("you need to 'pip install onnx' to valid the onnx file,\
                    it's ok if you choose not to check")
        net = onnx.load(onnx_file)
        onnx.checker.check_model(net)
        onnx.helper.printable_graph(net.graph)
        logger.success("onnxfile {} check finish", onnx_file)
        # valid onnx runtime
        try:
            import onnxruntime as rt
        except:
            logger.warning("cannot import onnxruntime, cannot valid inference")
            return onnx_file
        sess = rt.InferenceSession(onnx_file)
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: dummy_input.numpy()})[0]
        logger.debug("pred_onx : {}", pred_onx)
        logger.success("diff is {} %",
                (abs(pred_onx - output.numpy())).mean() * 100,
                )

        return onnx_file
