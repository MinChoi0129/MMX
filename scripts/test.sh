#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python predict.py --checkpoint ./logs/train/model_weights/model_best.pt