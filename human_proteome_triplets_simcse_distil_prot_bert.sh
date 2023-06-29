#!/bin/bash
#SBATCH --job-name=protein_benchmark_lpiekarski-nlp
#SBATCH --partition=nlp
#SBATCH --qos=1gpu1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output_human_proteome_triplets_simcse_distil_prot_bert.log
#SBATCH --error=error_human_proteome_triplets_simcse_distil_prot_bert.log

./venv/bin/python human_proteome_triplets/run.py lpiekarski/simcse_distil_prot_bert 8