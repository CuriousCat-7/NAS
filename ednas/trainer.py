import torch
import torch.nn as nn
from torch.nn.distributed import DataParallel
import torch.nn.functional as F

import time
import logging

from utils import AvgrageMeter, weights_init, \
                  CosineDecayLR
from utils import ModelTools
from flops_counter import *


class Trainer(object):
  """Training network parameters and theta separately.
  """
  def __init__(self, network,
               w_lr=0.01,
               w_mom=0.9,
               w_wd=1e-4,
               init_temperature=5.0,
               temperature_decay=0.965,
               target_lat=None,
               eta=0.1,
               lmd=1,
               logger=logging,
               lr_scheduler={'T_max' : 200},
               gpus=[0],
               model_save_path="",
               save_theta_prefix=''):
    assert isinstance(network, FBNet)
    network.apply(weights_init)
    network = network.train().cuda()
    if isinstance(gpus, str):
      gpus = [int(i) for i in gpus.strip().split(',')]
    network = DataParallel(network, gpus)
    self.gpus = gpus
    self._mod = network
    theta_params = network.theta
    mod_params = network.parameters()
    self.theta = theta_params
    self.w = mod_params
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    self.save_theta_prefix = save_theta_prefix
    self.eta = eta
    self.target_lat = target_lat
    self.lmd = lmd
    self.model_save_path = model_save_path

    self._acc_avg = AvgrageMeter('acc')
    self._ce_avg = AvgrageMeter('ce')
    self._lat_avg = AvgrageMeter('lat')
    self._loss_avg = AvgrageMeter('loss')
    self._valid_acc = []

    self.opt = torch.optim.SGD(
                    mod_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)

    self.sche = CosineDecayLR(self.opt, **lr_scheduler)

    self.base_lat = None
    self.input_shape = None

  def get_lat_loss(self, lat):
      if lat < self.target_lat * (1-self.eta):
          return -lat.log()
      elif lat > self.target_lat * (1+self.eta):
          return -lat.log()
      else:
          return lat * 0

  def build_base_lat(self, input_res):
    """
    base_lat = all flops - Conv2dMetaKernel's flops
    """
    assert len(input_res) >= 2
    flops_model = add_flops_counting_methods(self._mod)
    flops_model.eval()
    flops_model.start_flops_count()
    batch = torch.ones(()).new_empty(
        (1, *input_res),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)
    _ = flops_model(batch)
    flops_count = flops_model.compute_average_flops_cost()
    flops_meta = 0
    for m in self._mod.modules():
        if isinstance(m, Conv2dMetaKernel):
            for sub_conv in m.modules():
                if is_supported_instance(module):
                    flops_meta += sub_conv.__flops__
    self.base_lat = (flops_count - flops_meta).item()
    print(f"total_lat is {flops_count}",
          f"{flops_meta is {flops_meta}",
          f"base_lat is {self.base_lat}")

  def assert_input_shape(self, input_shape):
      if self.input_shape is None:
          self.input_shape = input_shape
      elif self.input_shape != input_shape:
          raise ValueError(
              "input_shape {input_shape} !=\
               per input_shape {self.input_shape}")

  def train_wt(self, input, target, decay_temperature=False):
    """Update model and theta parameters.
    """
    self._mod.train()
    self.assert_input_shape(input.shape)
    self.opt.zero_grad()
    if self.base_lat is None:
        self.build_base_lat(input.shape)
    logits = self._mod(input, target)
    ce = self.criterion(pred, target)
    lat = self._mod.get_lat() + self.base_lat
    lat_loss = self.get_lat_loss(lat)
    loss = ce + self.lmd * lat_loss
    batch_size = len(target)
    pred = torch.argmax(logits, dim=1)
    acc = torch.sum(pred == target).float() / batch_size

    self.opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat_loss.item(), acc.item()

  def decay_temperature(self, decay_ratio=None):
    tmp = self.temp
    if decay_ratio is None:
      self.temp *= self._tem_decay
    else:
      self.temp *= decay_ratio
    self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))

  def _step(self, input, target,
            epoch, step,
            log_frequence,
            func):
    """Perform one step of training.
    """
    input = input.cuda()
    target = target.cuda()
    loss, ce, lat, acc = func(input, target)

    # Get status
    batch_size = self._mod.batch_size

    self._acc_avg.update(acc)
    self._ce_avg.update(ce)
    self._lat_avg.update(lat)
    self._loss_avg.update(loss)

    if step > 1 and (step % log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)

      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s"
              % (epoch, step, speed, self._loss_avg,
                 self._acc_avg, self._ce_avg, self._lat_avg))
      map(lambda avg: avg.reset(), [self._loss_avg, self._acc_avg,
                                    self._ce_avg, self._lat_avg])
      self.tic = time.time()

  def search(self, train_w_ds,
            train_t_ds,
            total_epoch=90,
            start_w_epoch=10,
            valid_interval=1,
            log_frequence=100):
    """Search model.
    """
    assert start_w_epoch >= 1, "Start to train w"
    self._mod.set_temperature(self.temp)
    self.tic = time.time()
    for epoch in range(total_epoch):
      self.logger.info("Start to train w for epoch %d" % epoch)
      self.logger.info("warming up for until" % start_w_epoch)
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch,
                   step, log_frequence,
                   lambda x, y: self.train_wt(x, y, False))
        self.sche.step()
        if epoch >= start_w_epoch:
          self.decay_temperature()
          if epoch % valid_interval == 0:
            acc = self.valid(train_t_ds)
            if max(self.valid_acc) < acc:
              self.save(name=f"EP{epoch}_ACC{acc*100}.pth")
            self.valid_acc.append(acc)
            logger.info(f"acc curve {self.valid_acc}")
            logger.info(f"acc {acc}")
            logger.info(f"max_acc {max(self.valid_acc)}")

  def valid(self, train_t_ds):
    self._mod.eval()
    correct = 0
    total_num = 0
    for idx, (input, target) in enumerate(train_t_ds):
      input = input.cuda()
      target = target.cuda()
      pred = self._mod(input).argmax(dim=1)
      total_num += len(pred)
      correct += (pred == target).int().sum()
    acc = float(correct) / total_num
    return acc

  def save(self, name=None):
      path = f"{self.model_save_path}/{name}"
      ModelTools.save_model(self._mod, path+".pth")
