# FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy

## Introduction

This repository contains the code for **FD-MobileNet** (*Fast-Downsampling* MobileNet), an efficient and accurate network for very limited computational budgets (e.g., 10-140 MFLOPs).

Our paper *FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy* is under review as a conference paper at ICIP 2018. 

For details, please refer to our [arXiv version](https://arxiv.org/abs/1802.03750).

## Requirements

1. pytorch >= 0.2.0, torchvision >= 0.2.0
2. graphviz >= 0.8.0

## Usage

To train a model:

```bash
python -u main.py \
       --data /path/to/data/config \
       --model /path/to/model/config \
       --optim /path/to/optim/config \
       --sched /path/to/sched/config \
       --label model_label \
       [--print-freq N] \
       [--resume] \
       [--evaluate]
```

where `model_label` is the name of the checkpoint to be saved or resumed. For example:

```bash
python -u main.py \
       --data config/imagenet/data-config/imagenet-aggressive.json \
       --model config/imagenet/model-config/fd-mobilenet/1x-FDMobileNet-224.json \
       --optim config/imagenet/optim-config/SGD-120-nesterov.json \
       --sched config/imagenet/sched-config/StepLR-30-0.1.json \
       --label 1x-FDMobileNet-224
```

For simplicity, we train models and save checkpoints in multi-GPU models (using `torch.nn.DataParallel`), which means the keys in the `state_dict` saved have the prefix `module.`. To convert a multi-GPU model to single-GPU model, run `convert_model.py`:

```bash
python -u convert_model.py \
       --data /path/to/data/config \
       --model /path/to/model/config \
       --label model_label \
       --input /path/to/checkpoint/file \
       --output /path/to/output/file
```

Our pre-trained models are single-GPU models (without prefix). To evaluate single-GPU models, run `evaluate.py`:

```bash
python -u evaluate.py \
       --data /path/to/data/config \
       --model /path/to/model/config \
       --checkpoint /path/to/checkpoint/file \
       [--print-freq N]
```

`main.py` is modified from [the pytorch example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

## Results on ImageNet 2012

Training configuration:

1. 120 epochs;
2. Batch size: 256;
3. Momentum: 0.9;
4. Weight decay: 4e-5;
5. Learning rate starts from 0.1, divided by 10 every 30 epochs;
6. 4 GPUs.

| Model | MFLOPs| Top-1 Acc. (%) |
| --- | --- | --- |
| FD-MobileNet 1$\times$ | 144 | 65.3 |
| FD-MobileNet 0.5$\times$ | 40 | 56.3 |
| FD-MobileNet 0.25$\times$ | 12 | 45.1 |


