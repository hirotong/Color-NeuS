# !/usr/bin/bash
GPU=$1
cases=("insect5" "insect6" "insect7")

for case in ${cases[@]}; do
    python train.py -g $GPU --cfg config/Color_NeuS_iqscan_ref_pose.yml -obj jul13/$case --exp_id Color_NeuS_iqscan_refine_pose_$case
done