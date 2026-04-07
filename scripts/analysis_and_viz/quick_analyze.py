#!/usr/bin/env python3
"""
Quick analysis script for LLM dataset inference results.
Simple examples of how to use the analyze_single_file function.
"""

from pathlib import Path
from simple_visualize import analyze_single_file, analyze_all_files_in_directory

def quick_analyze_file(file_path):
    """Quick analysis of a single file - shows summary stats only."""
    print(f"Quick analysis of: {Path(file_path).name}")
    result = analyze_single_file(file_path, summary_only=True)
    return result

def quick_analyze_all(results_dir="/scratch/ar7789/llm_dataset_inference/results"):
    """Quick analysis of all files - shows summary stats only."""
    print(f"Quick analysis of all files in: {results_dir}")
    all_results = analyze_all_files_in_directory(results_dir, summary_only=True)
    return all_results

def analyze_specific_metrics(file_path, metrics=['ppl', 'zlib_ratio']):
    """Analyze specific metrics from a file."""
    print(f"Analyzing metrics {metrics} from: {Path(file_path).name}")
    result = analyze_single_file(file_path, metrics_to_analyze=metrics, summary_only=True)
    return result

# Example usage
if __name__ == "__main__":
    # Example 1: Analyze a specific file
    specific_file = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-160m-deduped/philpapers_train_metrics.json"
    
    if Path(specific_file).exists():
        print("=== Analyzing specific file ===")
        result = quick_analyze_file(specific_file)
    
    # Example 2: Analyze all files
    print("\n=== Analyzing all files ===")
    all_results = quick_analyze_all()
    
    # Example 3: Analyze specific metrics
    if Path(specific_file).exists():
        print("\n=== Analyzing specific metrics ===")
        result = analyze_specific_metrics(specific_file, ['ppl', 'zlib_ratio', 'k_min_probs_0.1'])
