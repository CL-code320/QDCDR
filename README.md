This source code is for the paper: “Cross-domain Recommendation via Quantized Disentangled Generative Model” submited to Information Fusion by Yi Zhao,Le Chen, Jibing Gong and et.al.

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Usage
---

To run this project, please make sure that you have the following packages being downloaded. 

Running example:

```shell

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --model QDCDR --dataset Sport_Cloth --id SC --num_epoch 500 > ./logs/Sport_Cloth.log 2>&1&

```

''''
