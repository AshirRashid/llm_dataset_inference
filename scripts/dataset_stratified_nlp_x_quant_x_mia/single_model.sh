#!/bin/bash
#SBATCH -o /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_nlp_%j.log
#SBATCH -e /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/slurm_out/slurm_nlp_%j.log
#SBATCH -t 50:00:00
#SBATCH --mem=64G

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

conda activate viz
cd /scratch/ar7789/llm_dataset_inference/scripts/dataset_stratified_nlp_x_quant_x_mia

python single_model.py
