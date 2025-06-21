This source code is for the paper: “Cross Domain Recommendation via Adaptive Bi-directional Transfer Graph Neural Networks” submited to Knowledge and Information Systems by Yi Zhao,Jinxin Ju Jibing Gong and et.al.

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a Tesla V100 32GB GPU. 

Running example:

```shell

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --model QDCDR --dataset Sport_Cloth --id SC --num_epoch 500 > ./logs/Sport_Cloth.log 2>&1&

```

''''
