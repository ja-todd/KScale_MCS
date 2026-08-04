#!/bin/bash

source ~/miniforge3/bin/activate
conda activate hk26_env

MODELS=(um_glm_n2560_RAL3p3_tuned_hk26 um_glm_n2560_CoMA9_hk26 um_glm_n1280_GAL9_v2_hk26 um_glm_n1280_CoMA9_hk26)

for model in "${MODELS[@]}"; do 
	time python ../build_1h_norm_lifecycle_dataset.py --model "$model"
done 