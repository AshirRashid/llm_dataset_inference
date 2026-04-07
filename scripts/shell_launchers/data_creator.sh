#!/bin/sh
#SBATCH --output=/scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/%j.out  # Standard output
#SBATCH --error=/scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/%j.err   # Standard error
#SBATCH --cpus-per-task=1             # Adjust as needed
#SBATCH --mem=8G                      # Adjust memory as needed
#SBATCH --time=2:00:00               # Adjust time limit as needed

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate llm_dataset_inference2


cd ..

for dataset in stackexchange wikipedia cc github pubmed_abstracts openwebtext2 freelaw math nih uspto hackernews enron books3 pubmed_central gutenberg arxiv bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers
do
    echo "dataset: $dataset"
    python data_creator.py --dataset_name $dataset 
done

