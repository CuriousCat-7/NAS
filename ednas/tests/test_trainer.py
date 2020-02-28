import os
import sys
sys.path.insert(0, os.getcwd())
import unittest
import torch
from torch import nn
import torchvision.datasets as dset

import numpy as np
import logging
import argparse
import time
import os

from model import EDNetV2 as EDNet
from trainer import Trainer
from data import get_ds
from utils import _logger, _set_file
from torch.utils.data.sampler import SubsetRandomSampler
from easydict import EasyDict as edict



class Config(object):
  num_cls_used = 0
  init_theta = 1.0
  alpha = 0.2
  beta = 0.6
  speed_f = './speed_cpu.txt'
  w_lr = 0.65 #0.1
  w_mom = 0.9
  w_wd = 3e-5 #1e-4
  t_lr = 0.01
  t_wd = 5e-4
  t_beta = (0.9, 0.999)
  init_temperature = 5.0
  temperature_decay = 0.956
  model_save_path = '/data/limingyao/model/nas/ednas/'
  total_epoch = 90
  start_w_epoch = 2
  train_portion = 0.8
  width_mult = 0.75
  kernel_size = [3, 5, 7]
  target_lat = 260e3
  valid_interval = 1

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
  'lr_mul' : 0.98,
}


class Test(unittest.TestCase):

    def build_trainer(self, config, args):
        if not os.path.exists(args.model_save_path):
          _logger.warn("{} not exists, create it".format(args.model_save_path))
          os.makedirs(args.model_save_path)
        _set_file(args.model_save_path + 'log.log')

        import torchvision.transforms as transforms
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
          ])


        train_data = dset.CIFAR10(root='/data/limingyao/.torch/datasets', train=True,
                        download=False, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(config.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=16)

        val_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=SubsetRandomSampler(indices[split:]),
          pin_memory=True, num_workers=8)

        self.train_queue = train_queue
        self.val_queue = val_queue

        model = EDNet(num_classes=config.num_cls_used if config.num_cls_used > 0 else 10,
                      width_mult=config.width_mult,
                      kernel_size=config.kernel_size)
        self.trainer = Trainer(network=model,
                  w_lr=config.w_lr,
                  w_mom=config.w_mom,
                  w_wd=config.w_wd,
                  init_temperature=config.init_temperature,
                  temperature_decay=config.temperature_decay,
                  logger=_logger,
                  lr_scheduler=lr_scheduler_params,
                  target_lat=config.target_lat,
                  gpus=args.gpus,
                  model_save_path=args.model_save_path)


    def test_build_lat(self):
        config = Config()
        args = edict()
        args.gpus = "0"
        args.model_save_path = '/tmp/debug_ednas'
        args.batch_size = 64
        config.width_mult = 0.75
        config.kernel_size= [3,5,7]

        self.build_trainer(config, args)

        trainer = self.trainer
        trainer.build_base_lat([1, 3, 224, 224])
        print(trainer.base_lat)
        trainer.cal_lat_range([1, 3, 224, 224])

    def test_search(self):
        config = Config()
        args = edict()
        args.gpus = "0"
        args.model_save_path = '/tmp/debug_ednas'
        args.batch_size = 64
        args.log_frequence = 100
        config.width_mult = 0.75
        config.kernel_size= [3,5,7]
        config.start_w_epoch = 1
        config.total_epoch = 10
        config.w_lr = 0.01
        self.build_trainer(config, args)

        trainer = self.trainer
        trainer.search(self.train_queue, self.val_queue,
               total_epoch=config.total_epoch,
               start_w_epoch=config.start_w_epoch,
               valid_interval=config.valid_interval,
               log_frequence=args.log_frequence)



if __name__ == "__main__":
    unittest.main()
