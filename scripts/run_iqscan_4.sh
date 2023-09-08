# !/usr/bin/bash
GPU=$1
cases=("insect7")

for case in ${cases[@]}; do
    python train.py -g $GPU --cfg config/Color_NeuS_iqscan_small.yml -obj jul13/$case --exp_id Color_NeuS_iqscan_$case
done
