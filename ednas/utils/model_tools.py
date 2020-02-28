from .dist_op import Dist
import torch
from loguru import logger
from collections import OrderedDict
from copy import deepcopy
import time


class ModelTools(object):
    @staticmethod
    def build_model(modelclass, resume, *args, force_random_init=False, **kwargs):
        if force_random_init:
            pretrained = False
        else:
            pretrained = False if resume else True

        model = modelclass(*args, pretrained=pretrained, **kwargs)

        if resume:
            model.load_state_dict(torch.load(resume,
                                             map_location=lambda storage,
                                             loc: storage))
        return model

    @classmethod
    def weights_to_cpu(cls, state_dict):
        """Copy a model state_dict to cpu.
        Args:
            state_dict (OrderedDict): Model weights on GPU.
        Returns:
            OrderedDict: Model weights on GPU.
        """
        state_dict_cpu = OrderedDict()
        for key, val in state_dict.items():
            state_dict_cpu[key] = val.cpu()
        return state_dict_cpu

    @staticmethod
    def get_timestamp():
        return time.strftime("%Y_%m_%d_%H_%M_%S")

    @classmethod
    def save_model(cls, model,  name):
        data_parallel = Dist.is_parallel(model)
        state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        torch.save( cls.weights_to_cpu(state_dict), name )
        logger.success("save {} finish", name)

    @classmethod
    def save_full_model(cls, model,  name):
        data_parallel = Dist.is_parallel(model)
        model = model.module if data_parallel else model
        model = deepcopy(model)
        model.cpu()
        torch.save(model, name)
        logger.success("save {} finish", name)

    @staticmethod
    def get_optim_sched(model, lr):
        optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=100, eta_min=lr*1e-2, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=1, T_mult=2)

        return optimizer, scheduler
