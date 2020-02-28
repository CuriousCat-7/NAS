#!/usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import sys
import os
import subprocess
from datetime import datetime


# for convert onnx

class Prepare():
    @staticmethod
    def vis_env():
        logger.info("turning on visdom server ")
        subprocess.call(
            "nohup visdom -port 8097 >/dev/null 2>&1 &", shell=True)

    @staticmethod
    def save_env():
        logger.info("making save dir")
        os.makedirs("./save", exist_ok=True)

    @classmethod
    def env(cls):
        cls.save_env()
        cls.save_env()

    @staticmethod
    def vis(env="main", port=8097):
        from .visualize import VisdomProxy
        from visdom import Visdom
        from concurrent.futures import ThreadPoolExecutor
        from .visualize import VisdomProxy
        if env is None:
                return None, None
        vis = Visdom(use_incoming_socket=False, env=env, port=port)
        executor = ThreadPoolExecutor(max_workers=4)
        proxy = VisdomProxy(vis, executor=executor)
        return vis, proxy


    @staticmethod
    def mlflow(mlflow_uri=None, tags={}, show_args={}):
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.error('ERROR: Failed importing mlflow')
            MlflowClient = NotImplemented

        if mlflow_uri is None:
            return None, None

        tags["git_branch"] = subprocess.check_output(["git", "describe", "--all"]).decode("utf8").strip()
        tags["git_hash"] = subprocess.check_output(["git", "describe", "--always"]).decode("utf8").strip()
        tags["timestamp"] = datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        if "MY_HOSTIP" in os.environ.keys():
            tags["my_host_ip"] = os.environ["MY_HOSTIP"]
        if "MY_HOSTNAME" in os.environ.keys():
            tags["my_host_name"] = os.environ["MY_HOSTNAME"]

        for key, value in tags.items():
            show_args[f'tags-{key}'] = value

        if "exp_name" in tags:
            exp_name = tags["exp_name"]
        else:
            exp_name = "Ocr-PSEOnly"

        ml_client = MlflowClient(mlflow_uri)
        ml_exp = ml_client.get_experiment_by_name(exp_name)
        if ml_exp is None:
            ml_exp_id = ml_client.create_experiment(exp_name)
        else:
            ml_exp_id = ml_exp.experiment_id
        ml_run = ml_client.create_run(ml_exp_id)
        logger.info("ml_run: {}", ml_run)
        ml_run_id = ml_run.info.run_id
        for key, value in tags.items():
            logger.info(f'tag --- {key}: {value}')
            if ml_client is not None:
                ml_client.set_tag(ml_run_id, key,value)
        logger.info("------------")
        for key, value in show_args.items():
            logger.info(f'{key}: {value}')
            if ml_client is not None:
                ml_client.log_param(ml_run_id, key,value)

        return ml_client, ml_run_id


# from hand_gesture
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        new_tensor = tensor.clone()
        for t, m, s in zip(new_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return new_tensor


class LoggerTools():

    @staticmethod
    def set_level():
        # https://github.com/Delgan/loguru/issues/138
        logger.remove()
        logger.add(sys.stderr, level="INFO")
