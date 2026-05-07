import json
import os

# Set HuggingFace cache directories to avoid permission issues with default root paths
os.environ['HF_HOME'] = '/scratch/ar7789/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/scratch/ar7789/.cache/huggingface/datasets'

# Limit multi-threading to avoid RLIMIT_NPROC issues on the cluster
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# Absolute path to current directory for robust imports
sys.path.append('/scratch/ar7789/llm_dataset_inference/scripts/dataset_x_nlp')
from datasets import load_dataset
from nlp_corpus_metrics import compute_corpus_metrics

def main():
    # The four primary datasets discussed in the multivariate analysis
    datasets_to_process = ["arxiv", "wikipedia", "github", "cc"]
    all_results = {}

    print("Starting NLP Metrics computation for all discussed datasets...")
    
    for ds_name in datasets_to_process:
        print(f"\n--- Dataset: {ds_name.upper()} ---")
        try:
            # Load from HuggingFace
            hf_train = load_dataset("pratyushmaini/llm_dataset_inference", name=ds_name, split="train")
            hf_val = load_dataset("pratyushmaini/llm_dataset_inference", name=ds_name, split="val")
            
            # Select the first 2000 samples from each split to match the analysis scope
            train_texts = [row['text'] for row in hf_train.select(range(min(2000, len(hf_train))))]
            val_texts = [row['text'] for row in hf_val.select(range(min(2000, len(hf_val))))]
            
            full_corpus = train_texts + val_texts
            print(f"Total samples to process: {len(full_corpus)}")
            
            # Compute metrics
            metrics = compute_corpus_metrics(full_corpus)
            all_results[ds_name] = metrics
            
            # Print results for the current dataset
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<25}: {value:.4f}")
                
        except Exception as e:
            print(f"Error processing dataset {ds_name}: {e}")

    # Save the consolidated results to a JSON file
    output_dir = "/scratch/ar7789/llm_dataset_inference/scripts/dataset_x_nlp"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "multivariate_datasets_nlp_metrics.json")
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nConsolidated results saved to: {output_file}")

if __name__ == "__main__":
    main()
