#!/bin/bash

source ~/miniforge3/bin/activate 
conda activate hk26_env

MODELS=(um_glm_n2560_RAL3p3_tuned_hk26 um_glm_n2560_CoMA9_hk26 um_glm_n1280_GAL9_v2_hk26 um_glm_n1280_CoMA9_hk26)
RADII=(50 100 200)
SURFACES=(land ocean)

for model in "${MODELS[@]}"; do
    for radius in "${RADII[@]}"; do
        for surface in "${SURFACES[@]}"; do
                echo "Running model=$model radius=$radius surface=$surface"
                python compute_entrainment_rate.py --init --model "$model" --radius "$radius" --surface "$surface"
                python compute_entrainment_rate.py --run  --model "$model" --radius "$radius" --surface "$surface"
	    done
    done
done
