#!/bin/bash
my_time=$((10 + RANDOM % 150))
sleep $my_time
source ~/.bashrc
conda activate for_mljar
~/anaconda3/envs/for_mljar/bin/python ./A_AutoML_mljar_v9_classification_full_HIA_config_11_USERCONFIG_11.py
exit 0
