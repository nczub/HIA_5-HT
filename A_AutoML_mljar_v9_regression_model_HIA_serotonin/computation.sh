#!/bin/bash
my_time=$((10 + RANDOM % 150))
sleep $my_time
source ~/.bashrc
conda activate for_mljar
~/anaconda3/envs/for_mljar/bin/python ./A_AutoML_mljar_v9_regression_full_HIA_config_9_USERCONFIG_9.py
exit 0
