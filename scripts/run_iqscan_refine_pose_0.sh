# !/usr/bin/bash
GPU=$1
cases=("insect1" "insect2" "insect3" "insect4")

for case in ${cases[@]}; do
    python train.py -g $GPU --cfg config/Color_NeuS_iqscan_ref_pose.yml -obj jul13/$case --exp_id Color_NeuS_iqscan_refine_pose_$case
done
