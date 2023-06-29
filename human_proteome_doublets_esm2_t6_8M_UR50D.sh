#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output_human_proteome_doublets_esm2_t6_8M_UR50D.log
#SBATCH --error=error_human_proteome_doublets_esm2_t6_8M_UR50D.log
#SBATCH --time=0-3:00:00

./venv/bin/python human_proteome_doublets/run.py facebook/esm2_t6_8M_UR50D 8