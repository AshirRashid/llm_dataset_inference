#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_dynamic_ppl_%j.log
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_dynamic_ppl_%j.log
#SBATCH -t 10:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

# Load compiler module for Triton kernel compilation
module load gcc/9.2.0 2>/dev/null || module load gcc/13.2.0 2>/dev/null || true

# Initialize conda for bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate base
eval "$(conda shell.bash hook)"

# Activate the environment used for dynamic quantization
conda activate pytorch_quant

cd /scratch/ar7789/llm_dataset_inference/scripts

# Run the perplexity calculation script
python compute_dynamic_ppl.py
