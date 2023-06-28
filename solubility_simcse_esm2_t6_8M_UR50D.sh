#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output_solubility_simcse_esm2_t6_8M_UR50D.log
#SBATCH --error=error_solubility_simcse_esm2_t6_8M_UR50D.log

./venv/bin/python solubility/run.py lpiekarski/simcse_esm2_t6_8M_UR50D 8