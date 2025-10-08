#!/bin/bash

echo "[Pretraining...]"
python pretrain.py
echo "[Done...]"

echo "[Training...]"
python train.py
echo "[Done...]"

echo "[Predicting...]"
python predict.py
echo "[Done...]"