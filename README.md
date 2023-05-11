# LSAS: Lightweight Sub-attention Strategy for Alleviating Attention Bias Problem 
[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/Qrange%20-group-orange)

This repository is the implementation of "LSAS: Lightweight Sub-attention Strategy for Alleviating Attention Bias Problem" [[paper]](https://arxiv.org/abs/2305.05200) on CIFAR-100, CIFAR-10, STL10 and ImageNet datasets. Our paper has been accepted for presentation at ICME 2023.

## Introduction

In computer vision, the performance of deep neural networks (DNNs) is highly related to the feature extraction ability, i.e., the ability to recognize and focus on key pixel regions in an image. However, in this paper, we quantitatively and statistically illustrate that DNNs have a serious attention bias problem on many samples from some popular datasets: (1) Position bias: DNNs fully focus on label-independent regions; (2) Range bias: The focused regions from DNN are not completely contained in the ideal region. Moreover, we find that the existing self-attention modules can alleviate these biases to a certain extent, but the biases are still non-negligible. To further mitigate them, we propose a lightweight sub-attention strategy (LSAS), which utilizes high-order sub-attention modules to improve the original self-attention modules. The effectiveness of LSAS is demonstrated by extensive experiments on widely-used benchmark datasets and popular attention networks. 

<p align="center">
  <img src="https://github.com/Qrange-group/LSAS/blob/main/images/arch.png" width="600" height="300">
</p>


## Requirement
Python and [PyTorch](http://pytorch.org/).
```
pip install -r requirements.txt
```


## Usage
```sh
# run ResNet164-SENet on cifar10, 1 GPU
CUDA_VISIBLE_DEVICES=0 python run.py --arch senet --dataset cifar10 --block-name bottleneck --depth 164 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4

# run ResNet164-LSAS-SENet on cifar10, 1 GPU
CUDA_VISIBLE_DEVICES=0 python run.py --arch lsas_senet --dataset cifar10 --block-name bottleneck --depth 164 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4

# run ResNet50-SENet on ImageNet, 8 GPUs
python -u -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port='29503' run_imagenet.py -a senet_resnet50 --info normal --data /data1/ZSS/datasets/ILSVRC2012_Data --epochs 100 --schedule 30 60 90 --wd 1e-4 --gamma 0.1 --train-batch 32 --opt-level O0 --wd-all --label-smoothing 0. --warmup 0

# run ResNet50-LSAS-SENet on ImageNet, 8 GPUs
python -u -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port='29503' run_imagenet.py -a lsas_senet_resnet50 --info normal --data /data1/ZSS/datasets/ILSVRC2012_Data --epochs 100 --schedule 30 60 90 --wd 1e-4 --gamma 0.1 --train-batch 32 --opt-level O0 --wd-all --label-smoothing 0. --warmup 0
```

## Results
|                 |  Dataset  | SENet |  LSAS-SENet  |
|:---------------:|:------:|:--------:|:------:|
|    ResNet164    |CIFAR10 |   94.57  |  95.01 |
|    ResNet164    |CIFAR100|   75.30  |  76.47 |
|    ResNet164    |STL10   |   83.81  |  85.71 |
|    ResNet50     |ImageNet|   76.63  |  77.28 |



## Citation

```
@inproceedings{Zhong2023LSASLS,
  title={LSAS: Lightweight Sub-attention Strategy for Alleviating Attention Bias Problem},
  author={Shan Zhong and Wushao Wen and Jinghui Qin and Qiangpu Chen and Zhongzhan Huang},
  year={2023}
}
```

## Acknowledgments
Many thanks to [bearpaw](https://github.com/bearpaw) for his simple and clean [Pytorch framework](https://github.com/bearpaw/pytorch-classification) for image classification task.
