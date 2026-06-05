#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%A_%a.log
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%A_%a.log
#SBATCH -t 150:00:00
#SBATCH --mem=128G
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --array=0-73

# Array of models
models=(
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-awq-b4-gs128-zp1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-awq-b4-gs32-zp1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-awq-b4-gs64-zp1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs128-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs128-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs-1-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs-1-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs32-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs32-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs64-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b2-gs64-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs128-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs128-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs-1-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs-1-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs32-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs32-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs64-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b3-gs64-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs128-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs128-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs-1-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs-1-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs32-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs32-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs64-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs64-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs128-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs128-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs-1-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs-1-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs32-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs32-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs64-da0
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs64-da1
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-bnb4
    /scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-bnb8
)

# EleutherAI/pythia-1b-deduped
# EleutherAI/pythia-410m-deduped

# Array of datasets
# Original including commented ones: bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers pubmed_abstracts math nih enron stackexchange wikipedia cc github openwebtext2 freelaw uspto hackernews books3 pubmed_central gutenberg arxiv
# datasets=(bookcorpus2 books3 cc europarl freelaw github gutenberg hackernews math openwebtext2 opensubtitles philpapers stackexchange uspto ubuntu wikipedia youtubesubtitles arxiv)
# datasets=("wikipedia" "arxiv" "github" "cc")
datasets=("hackernews" "math")

# Total jobs = num_models * num_datasets
num_datasets=${#datasets[@]}
model_idx=$((SLURM_ARRAY_TASK_ID / num_datasets))
dataset_idx=$((SLURM_ARRAY_TASK_ID % num_datasets))

model_name=${models[$model_idx]}
dataset=${datasets[$dataset_idx]}

# Logic for batch size assignment
quant=""
if [[ "$model_name" =~ "pythia-12b" ]]; then
    echo "Model is pythia-12b"
    batch_size=1
    quant="fp16"
elif [[ "$model_name" =~ "pythia-6.9b-deduped" ]]; then
    echo "Model is pythia-6.9b-deduped"
    batch_size=4
elif [[ "$model_name" =~ "pythia-1b-deduped" ]]; then
    echo "Model is pythia-1b-deduped"
    batch_size=8
else
    batch_size=32
fi

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Batch Size: $batch_size"

# Execute the original individual launcher
# We call it with bash to run it as a script within this job allocation
# Arguments: model_name, gpu_id, batch_size, dataset, quant
# We pass "0" as gpu_id because quantized models fit comfortably on a single A100, and splitting across GPUs severely degrades bitsandbytes performance.
bash /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/custom_di_launcher_individual.sh "$model_name" "0" "$batch_size" "$dataset" "$quant"
