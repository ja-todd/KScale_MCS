#!/bin/bash
#SBATCH --account=mcs_prime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:0:00
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --job-name=mcs_entrainment
#SBATCH --output=slurm/output/%j.out
#SBATCH --error=slurm/output/%j.err

# while [[ "$#" -gt 0 ]]; do
#    case $1 in
#        --model) MODEL="$2"; shift ;;
#        --region) REGION="$2"; shift ;;
#    esac
#    shift
# done


source ~/miniforge3/bin/activate
conda activate hk26_env

python calc_mcs_entrainment.py --model um_glm_n1280_CoMA9_hk26 --region wam 
