# Efficient Differentiable Neural Architecture Search with Meta Kernels

Shoufa Chen, Yunpeng Chen, Shuicheng Yan, Jiashi Feng (2019)

Efficient Differentiable Neural Architecture Search with Meta Kernels

https://arxiv.org/abs/1912.04749

Stupid but straightforward nas method

## Train cifar10
```shell
python train_cifar10.py --batch-size 32 --log-frequence 100
```

## Train ImageNet
Randomly choose 100 classes from 1000.
You need specify the root dir `imagenet_root` of ImageNet in `train.py`.
```shell
python train.py --batch-size $[24*8] --log-frequence 100 --gpus 0,1,2,3,4,5,6,7
```


