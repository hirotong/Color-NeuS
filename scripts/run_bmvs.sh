# !/usr/bin/bash
conda activate diffusionsdf
GPU=$1

cases=("bear" "dog" "jade" "sculpture" "clock" "durian" "man" "stone")

for case in ${cases[@]}; do
    python train.py -g $GPU --cfg config/Color_NeuS_bmvs.yml -obj $case --exp_id Color_NeuS_bmvs_case
done
