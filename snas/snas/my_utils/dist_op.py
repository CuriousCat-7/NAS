import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel
from torch.nn.parallel import gather
from loguru import logger
import random
import os


class Dist():
    @staticmethod
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        for i in range(100):
            port = 12355 + random.randint(0, 20000)
            try:
                os.environ['MASTER_PORT'] = str(port)
            except:
                logger.warning(
                    "port {} cannot be used now, try to kill -9 zombie processes", port)
                continue

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        #dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @classmethod
    def ddp(cls, model, dim=0):
        if torch.cuda.device_count() > 1:
            logger.info("using {} gpus", torch.cuda.device_count())
            cls.setup(0, 1)
            model = DistributedDataParallel(model, dim=dim, find_unused_parameters=True)
            model.gather_dim = dim
        return model

    @staticmethod
    def dp(model, dim=0):
        if torch.cuda.device_count() > 1:
            logger.info("using {} gpus", torch.cuda.device_count())
            model = DataParallel(model, dim=dim)
        return model

    @staticmethod
    def ddp_end(cls):
        if torch.cuda.device_count() > 1:
            cls.cleanup()

    class DDP(DistributedDataParallel):
        def __init__(self, *args, **kwargs):
            super(DDP, self).__init__(*args, **kwargs)
            self.gather_dim = self.dim

        def gather(self, outputs, output_device):
            return gather(outputs, output_device, dim=self.gather_dim)

    class DP(DataParallel):
        def __init__(self, *args, **kwargs):
            super(DP, self).__init__(*args, **kwargs)
            self.gather_dim = self.dim

        def gather(self, outputs, output_device):
            return gather(outputs, output_device, dim=self.gather_dim)

    @classmethod
    def new_ddp(cls, model, dim):
        if torch.cuda.device_count() > 1:
            logger.info("using {} gpus", torch.cuda.device_count())
            cls.setup(0, 1)
            model = cls.DDP(model)
            model.gather_dim = dim
        return model

    @classmethod
    def new_dp(cls, model, dim=0):
        if torch.cuda.device_count() > 1:
            logger.info("using {} gpus", torch.cuda.device_count())
            model = cls.DP(model)
            model.gather_dim = dim
        return model

    @classmethod
    def is_parallel(cls, model):
        if isinstance(
                model,
                (DistributedDataParallel, DataParallel, cls.DP, cls.DDP)):
            return True
        else:
            return False
