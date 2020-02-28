# encoding: UTF-8

import copy
import math
import torch
import visdom
import inspect
import functools
import numpy as np


class VisdomProxy:
    def __init__(self, client, executor):
        self.client = client
        self.executor = executor

    @staticmethod
    def _to_tensor(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() is 0:
            x = x.view(1)
        if x.requires_grad:
            x = x.detach()
        return x

    def _add_default_title(self, win, kwargs):
        opts = kwargs.get('opts', {})
        if 'title' not in opts and win is not None:
            opts['title'] = win
        kwargs['opts'] = opts
        return kwargs

    def plot(self, x, y, win=None, **kwargs):
        update = 'append' if self.win_exists(win) else None
        x, y = self._to_tensor(x), self._to_tensor(y)
        kwargs = self._add_default_title(win, kwargs)
        return self.client.line(X=x, Y=y, win=win, update=update, **kwargs)

    def text(self, text, win=None, append=True, **kwargs):
        kwargs = self._add_default_title(win, kwargs)
        return self.client.text(
            text, win=win, append=append and self.win_exists(win),
            **kwargs)

    def grid(self, *args, **kwargs):
        self.executor.submit(self._grid,*args,  **kwargs)

    def _grid(self, tensor, win=None, **kwargs):
        if torch.is_tensor(tensor):
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
        if isinstance(tensor, list):
            tensor = [np.array(t) for t in tensor]
        kwargs = self._add_default_title(win, kwargs)
        return self.client.images(tensor, win=win, **kwargs)

    def imshow(self, tensor, win=None, **kwargs):
        if torch.is_tensor(tensor):
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
        kwargs = self._add_default_title(win, kwargs)
        return self.client.image(tensor, win=win, **kwargs)


def visualizer(vis, args, x, preds, rboxes, thetas, score):
    if x.get_device() != 0:
        return
    for i, (pred, rbox, theta) in enumerate(zip(preds, rboxes, thetas)):
        r = 2 ** i
        vis.grid(pred.detach().clone(), win='Prediction/{}'.format(i))
        vis.grid((score[:, :, ::r, ::r] * rbox / args.crop_size)[:, :1].detach().clone(), win=f'Approximate/{i}')
        # theta [-pi/4, pi/4] ->

