#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_nlp_metrics_%j.log
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_nlp_metrics_%j.log
#SBATCH -t 04:00:00
#SBATCH --mem=32G
#SBATCH -c 4

# Initialize conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate base
eval "$(conda shell.bash hook)"
conda activate viz

# Load GCC to provide newer libstdc++
module load gcc/13.2.0
export LD_LIBRARY_PATH=/share/apps/NYUAD/gcc/13.2.0/lib64:$LD_LIBRARY_PATH

cd /scratch/ar7789/llm_dataset_inference
python scripts/dataset_x_nlp/run_nlp_metrics_all_datasets.py
