#!/bin/bash

# Base directory for models
MODEL_BASE="/scratch/ar7789/llm_quant/saved_qmodels/EleutherAI"

# List of models (names derived from user request, stripped of .yaml)
MODELS=(
    # "pythia-160m-deduped-awq-b4-gs32-zp1-vGEMM"
    # "pythia-160m-deduped-dynamic-8bit"
    # "pythia-160m-deduped-gptq-b4-gs128-a0-d0.01"
    # "pythia-160m-deduped-gptq-b8-gs-1-a1-d0.01"
    "pythia-160m-deduped-static-4bit-fp4-bfloat16"
    "pythia-160m-deduped-static-8bit-ns256"
)

# Loop through models and submit jobs
for model in "${MODELS[@]}"; do
    FULL_MODEL_PATH="${MODEL_BASE}/${model}"
    
    # Check if model directory exists
    if [ ! -d "$FULL_MODEL_PATH" ]; then
        echo "WARNING: Model directory not found: $FULL_MODEL_PATH"
        # Optional: continue or exit? I'll continue to try others.
        # continue 
    fi

    echo "Submitting job for model: $model"
    # Assuming standard arguments for dataset etc. Adjust if needed.
    # defaulting to "wikipedia" and "train" (though script runs both splits now)
    sbatch /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/custom_di_launcher_individual.sh "$FULL_MODEL_PATH" "train" "0" "32" "github"
    
    echo "-----------------------------------"
done
