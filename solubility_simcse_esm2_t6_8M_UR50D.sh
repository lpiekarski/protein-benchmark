#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=results/simcse_esm2_t6_8M_UR50D/solubility/output.log
#SBATCH --error=results/simcse_esm2_t6_8M_UR50D/solubility/error.log

./venv/bin/python solubility/run.py lpiekarski/simcse_esm2_t6_8M_UR50D 1