#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.log
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_%j.log
#SBATCH -t 50:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

# Get command line arguments with defaults
model_name=${1:-"/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-410m-deduped-gptq-b8-gs128-da1"}
gpu_id=${2:-0}
batch_size=${3:-32}
dataset=${4:-"github"}
quant=${5:-""}

# Create descriptive filename components
timestamp=$(date +%Y%m%d_%H%M%S)
timestamp_readable=$(date '+%H:%M:%S_%Y-%m-%d_%Z')
job_suffix="${timestamp_readable}_${model_name}_${dataset}_bs${batch_size}_gpu${gpu_id}"

# Create human-readable directory if it doesn't exist
# mkdir -p /scratch/ar7789/llm_dataset_inference/scripts/human_slurm_out

export HF_HOME=/scratch/ar7789/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ar7789/.cache/huggingface

# Load compiler module for Triton kernel compilation (needed for AWQ)
# Try GCC 9.2.0 first (has stddef.h), fallback to 13.2.0
module load gcc/9.2.0 2>/dev/null || module load gcc/13.2.0 2>/dev/null || true

# Set CC and CXX environment variables for Triton
if command -v gcc &> /dev/null; then
    export CC=$(which gcc)
    export CXX=$(which g++ 2>/dev/null || which gcc)
    
    # Find GCC's include directories including builtin headers
    # Get GCC installation directory and version
    GCC_BIN=$(which gcc)
    GCC_DIR=$(dirname $(dirname "$GCC_BIN"))
    GCC_VERSION=$(gcc -dumpversion 2>/dev/null || echo "")
    
    # Build comprehensive include path
    # Priority: GCC 9.2.0 headers first (has stddef.h), then current GCC version, then system
    INCLUDE_PATH_LIST=""
    
    # Add GCC 9.2.0 headers FIRST (has stddef.h - this is critical!)
    GCC92_INCLUDE="/share/apps/NYUAD/gcc/9.2.0/lib/gcc/x86_64-pc-linux-gnu/9.2.0/include"
    if [ -d "$GCC92_INCLUDE" ] && [ -f "$GCC92_INCLUDE/stddef.h" ]; then
        INCLUDE_PATH_LIST="$GCC92_INCLUDE"
    fi
    
    # Add current GCC version's built-in headers if different from 9.2.0
    if [ -n "$GCC_VERSION" ]; then
        GCC_BUILTIN_INCLUDE="$GCC_DIR/$GCC_VERSION/lib/gcc/x86_64-pc-linux-gnu/$GCC_VERSION/include"
        if [ -d "$GCC_BUILTIN_INCLUDE" ] && [ "$GCC_BUILTIN_INCLUDE" != "$GCC92_INCLUDE" ]; then
            INCLUDE_PATH_LIST="${INCLUDE_PATH_LIST}${INCLUDE_PATH_LIST:+:}$GCC_BUILTIN_INCLUDE"
        fi
    fi
    
    # Add GCC's C++ headers if available
    if [ -n "$GCC_VERSION" ]; then
        for cppdir in "include/c++/$GCC_VERSION" "include/c++/$GCC_VERSION/x86_64-pc-linux-gnu"; do
            if [ -d "$GCC_DIR/$cppdir" ]; then
                INCLUDE_PATH_LIST="${INCLUDE_PATH_LIST}${INCLUDE_PATH_LIST:+:}$GCC_DIR/$cppdir"
            fi
        done
    fi
    
    # Extract GCC's default system include search paths
    GCC_INCLUDE_DIRS=$(gcc -E -x c - -v < /dev/null 2>&1 | sed -n '/^#include </,/^End of search list/p' | grep "^ /" | tr '\n' ':')
    if [ -n "$GCC_INCLUDE_DIRS" ]; then
        INCLUDE_PATH_LIST="${INCLUDE_PATH_LIST}${INCLUDE_PATH_LIST:+:}$GCC_INCLUDE_DIRS"
    fi
    
    # Add common system include directories
    for dir in /usr/include /usr/include/x86_64-linux-gnu /usr/local/include; do
        if [ -d "$dir" ]; then
            INCLUDE_PATH_LIST="${INCLUDE_PATH_LIST}${INCLUDE_PATH_LIST:+:}$dir"
        fi
    done
    
    # Set environment variables for Triton compilation
    if [ -n "$INCLUDE_PATH_LIST" ]; then
        export C_INCLUDE_PATH="$INCLUDE_PATH_LIST"
        export CPATH="$INCLUDE_PATH_LIST"
    fi
    
    # Also set LIBRARY_PATH for linking
    if [ -d "/usr/lib64" ]; then
        export LIBRARY_PATH="/usr/lib64:/lib64:$LIBRARY_PATH"
    fi
fi

# Initialize conda for bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate base
eval "$(conda shell.bash hook)"

# Determine conda environment based on model_name
if echo "$model_name" | grep -q "awq"; then
    echo "Activating conda environment: awq" >&2
    conda activate awq || { echo "ERROR: Failed to activate awq environment" >&2; exit 1; }
elif echo "$model_name" | grep -q "gptq"; then
    echo "Activating conda environment: gptq" >&2
    conda activate gptq || { echo "ERROR: Failed to activate gptq environment" >&2; exit 1; }
elif echo "$model_name" | grep -qE "static|dynamic"; then
    echo "Activating conda environment: pytorch_quant" >&2
    conda activate pytorch_quant || { echo "ERROR: Failed to activate pytorch_quant environment" >&2; exit 1; }
else
    echo "Activating conda environment: gptq" >&2
    conda activate gptq || { echo "ERROR: Failed to activate gptq environment" >&2; exit 1; }
    # echo "ERROR: No valid conda environment found for model_name: $model_name" >&2
fi

echo "Active conda environment: $CONDA_DEFAULT_ENV" >&2


cd /scratch/ar7789/llm_dataset_inference

# Display arguments in human-readable format
echo "Raw Args: $@" >&2
echo "=========================================="
echo "JOB ARGUMENTS (in order):"
echo "=========================================="
echo "1. model_name:    $model_name"
echo "2. gpu_id:        $gpu_id"
echo "3. batch_size:    $batch_size"
echo "4. dataset:       $dataset"
echo "=========================================="
echo "DERIVED PARAMETERS:"
echo "=========================================="
echo "model_short:      $model_short"
echo "job_suffix:       $job_suffix"
echo "timestamp:        $timestamp_readable"
echo "=========================================="

# Create descriptive symbolic links for easier identification
echo "Creating descriptive symlinks..."
# ln -sf "/scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_${SLURM_JOB_ID}.log" "/scratch/ar7789/llm_dataset_inference/scripts/human_slurm_out/${job_suffix}.log" 2>/dev/null || true

# Set thread limits to avoid OpenBLAS error
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running unquantized model (TRAIN split)..."
QUANT_ARG=""
if [ -n "$quant" ]; then
    QUANT_ARG="--quant $quant"
fi
CUDA_VISIBLE_DEVICES=$gpu_id python di.py --model_name "$model_name" --dataset_name "$dataset" --split "train" --num_samples 2000 --batch_size $batch_size --cache_dir "/scratch/ar7789/.cache/huggingface" $QUANT_ARG

echo "Running unquantized model (VAL split)..."
CUDA_VISIBLE_DEVICES=$gpu_id python di.py --model_name "$model_name" --dataset_name "$dataset" --split "val" --num_samples 2000 --batch_size $batch_size --cache_dir "/scratch/ar7789/.cache/huggingface" $QUANT_ARG

# Compute p-values if both train and val metrics exist
train_metrics="results/${model_name}/${dataset}_train_metrics.json"
val_metrics="results/${model_name}/${dataset}_val_metrics.json"

if [ -f "$train_metrics" ] && [ -f "$val_metrics" ]; then
    echo "Files found: $train_metrics and $val_metrics"
    echo "Calculating p-values..."
    # Use 500 samples for training so that the remaining 500 (from 1000 total) are used for calculation
    
    # Run with default metric (ttest)
    python linear_di.py --model_name "$model_name" --dataset_name "$dataset" --num_samples 500 --normalize train --outliers "mean+p-value" --features all --num_random 5 --false_positive 0 
    python linear_di.py --model_name "$model_name" --dataset_name "$dataset" --num_samples 500 --normalize train --outliers "mean+p-value" --features all --num_random 5 --false_positive 1
    
    # Run with cohens_d metric
    echo "Calculating cohens_d metric..."
    python linear_di.py --model_name "$model_name" --dataset_name "$dataset" --num_samples 500 --normalize train --outliers "mean+p-value" --features all --num_random 5 --false_positive 0 --metric cohens_d
    python linear_di.py --model_name "$model_name" --dataset_name "$dataset" --num_samples 500 --normalize train --outliers "mean+p-value" --features all --num_random 5 --false_positive 1 --metric cohens_d
else
    echo "P-value calculation skipped: waiting for both train and val metrics."
    echo "Checked paths:"
    echo "  Train: $train_metrics"
    echo "  Val:   $val_metrics"
fi
