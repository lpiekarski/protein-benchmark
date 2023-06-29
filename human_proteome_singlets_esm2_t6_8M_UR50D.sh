#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output_human_proteome_singlets_esm2_t6_8M_UR50D.log
#SBATCH --error=error_human_proteome_singlets_esm2_t6_8M_UR50D.log

./venv/bin/python human_proteome_singlets/run.py facebook/esm2_t6_8M_UR50D 8