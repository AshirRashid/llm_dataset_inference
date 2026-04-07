#!/usr/bin/env python3
"""
Script to analyze all results files in the results directory.
This demonstrates how to use the analyze_single_file function.
"""

import os
from pathlib import Path
from simple_visualize import analyze_single_file, analyze_all_files_in_directory, compare_metrics_across_files

def find_all_json_files(results_dir):
    """Find all JSON files in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return []
    
    json_files = list(results_path.rglob('*.json'))
    return [str(f) for f in json_files]

def analyze_single_file_example(file_path):
    """Example of how to analyze a single file."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE: Analyzing single file")
    print(f"{'='*80}")
    
    # Analyze a single file with all metrics
    result = analyze_single_file(file_path)
    
    if result:
        print(f"\nSuccessfully analyzed: {result['dataset_name']}")
        print(f"Available metrics: {len(result['available_metrics'])}")
        print(f"Analyzed metrics: {len(result['results'])}")
    
    return result

def analyze_specific_metrics_example(file_path, metrics):
    """Example of how to analyze specific metrics."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE: Analyzing specific metrics")
    print(f"{'='*80}")
    
    # Analyze specific metrics only
    result = analyze_single_file(file_path, metrics_to_analyze=metrics, summary_only=True)
    
    if result:
        print(f"\nSuccessfully analyzed: {result['dataset_name']}")
        print(f"Analyzed metrics: {list(result['results'].keys())}")
    
    return result

def analyze_all_files_example(results_dir):
    """Example of how to analyze all files in a directory."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE: Analyzing all files in directory")
    print(f"{'='*80}")
    
    # Analyze all files
    all_results = analyze_all_files_in_directory(results_dir, summary_only=True)
    
    if all_results:
        print(f"\nSuccessfully analyzed {len(all_results)} files")
        
        # Get all unique metrics across all files
        all_metrics = set()
        for result in all_results:
            all_metrics.update(result['available_metrics'])
        
        print(f"Total unique metrics found: {len(all_metrics)}")
        print(f"Metrics: {sorted(all_metrics)}")
        
        # Compare specific metrics across all files
        if 'ppl' in all_metrics:
            compare_metrics_across_files(all_results, 'ppl')
        
        if 'zlib_ratio' in all_metrics:
            compare_metrics_across_files(all_results, 'zlib_ratio')
    
    return all_results

def main():
    # Configuration
    results_dir = "/scratch/ar7789/llm_dataset_inference/results"
    
    print("LLM Dataset Inference Results Analysis")
    print("=" * 50)
    
    # Find all JSON files
    json_files = find_all_json_files(results_dir)
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for i, file_path in enumerate(json_files[:10]):  # Show first 10
        print(f"  {i+1}. {file_path}")
    if len(json_files) > 10:
        print(f"  ... and {len(json_files) - 10} more")
    
    # Example 1: Analyze a single file (first file found)
    if json_files:
        print(f"\n{'='*80}")
        print("EXAMPLE 1: Single file analysis")
        print(f"{'='*80}")
        
        first_file = json_files[0]
        result = analyze_single_file_example(first_file)
    
    # Example 2: Analyze specific metrics
    if json_files:
        print(f"\n{'='*80}")
        print("EXAMPLE 2: Specific metrics analysis")
        print(f"{'='*80}")
        
        # Analyze only perplexity and zlib_ratio
        specific_metrics = ['ppl', 'zlib_ratio']
        result = analyze_specific_metrics_example(first_file, specific_metrics)
    
    # Example 3: Analyze all files
    print(f"\n{'='*80}")
    print("EXAMPLE 3: All files analysis")
    print(f"{'='*80}")
    
    all_results = analyze_all_files_example(results_dir)
    
    # Example 4: Programmatic usage
    print(f"\n{'='*80}")
    print("EXAMPLE 4: Programmatic usage")
    print(f"{'='*80}")
    
    # You can also use the functions programmatically
    for file_path in json_files[:3]:  # Analyze first 3 files
        print(f"\nAnalyzing: {Path(file_path).name}")
        result = analyze_single_file(file_path, summary_only=True)
        
        if result and 'ppl' in result['results']:
            ppl_stats = result['results']['ppl']
            print(f"  PPL - Mean: {ppl_stats['mean']:.3f}, Std: {ppl_stats['std_dev']:.3f}")
        
        if result and 'zlib_ratio' in result['results']:
            zlib_stats = result['results']['zlib_ratio']
            print(f"  Zlib - Mean: {zlib_stats['mean']:.3f}, Std: {zlib_stats['std_dev']:.3f}")

if __name__ == "__main__":
    main()
