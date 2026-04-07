#!/bin/sh
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.out
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.err
#SBATCH -t 05:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

model_name=$1
split_name=$2
gpu_id=$3
batch_size=$4
dataset=$5

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate llm_dataset_inference2


# if [ $model_name = "kernelmachine/silo-pdswby-1.3b" ]
# then
#     conda activate di_silo
# else
#     conda activate di
# fi

cd /scratch/ar7789/llm_dataset_inference

echo "model_name: $model_name" split_name: $split_name gpu_id: $gpu_id

echo "dataset: $dataset"
CUDA_VISIBLE_DEVICES=$gpu_id python di.py --split $split_name --dataset_name $dataset --model_name $model_name --batch_size $batch_size