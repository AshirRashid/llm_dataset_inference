#!/bin/bash
#SBATCH -o analysis_%j.log
#SBATCH -e analysis_%j.log
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -c 4

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

# Initialize conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate base
eval "$(conda shell.bash hook)"
conda activate viz

# Run the unified analysis
cd /scratch/ar7789/llm_dataset_inference/scripts/dataset_stratified_nlp_x_quant_x_mia/all_models
python main.py

echo "Analysis complete. Results stored in analysis_results.json and report.txt"
