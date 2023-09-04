# !/usr/bin/bash
GPU=$1
cases=("105" "106" "110" "114" "118" "122" "24" "37" "40" "55" "63" "65" "69" "83" "97")

for case in ${cases[@]}; do
    python train.py -g $GPU --cfg config/Color_NeuS_dtu.yml -obj $case --exp_id Color_NeuS_dtu_$case
done
