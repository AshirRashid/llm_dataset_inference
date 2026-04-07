#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.out
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.err
#SBATCH -t 00:5:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate llm_dataset_inference2

cd /scratch/ar7789/llm_dataset_inference

python di.py --model_name "EleutherAI/pythia-160m-deduped" --dataset_name "wikipedia" --split "train" --num_samples 10 --batch_size 2
# python di.py --model_name "EleutherAI/pythia-160m-deduped" --dataset_name "wikipedia"
