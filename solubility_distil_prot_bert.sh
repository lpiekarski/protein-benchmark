#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output_solubility_distil_prot_bert.log
#SBATCH --error=error_solubility_distil_prot_bert.log
#SBATCH --time=0-3:00:00

./venv/bin/python solubility/run.py yarongef/DistilProtBert 8