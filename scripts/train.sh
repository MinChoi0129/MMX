#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --checkpoint ./logs/pretrain/model_weights/model_best.pt