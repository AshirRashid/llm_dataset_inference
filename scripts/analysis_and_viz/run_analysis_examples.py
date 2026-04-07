#!/usr/bin/env python3
"""
Comprehensive examples of how to use the analyze_single_file function.
This script demonstrates all the different ways to analyze your results.
"""

import os
from pathlib import Path
from simple_visualize import analyze_single_file, analyze_all_files_in_directory, compare_metrics_across_files

def example_1_single_file_analysis():
    """Example 1: Analyze a single file with all metrics and histograms."""
    print("=" * 80)
    print("EXAMPLE 1: Single file analysis with full details")
    print("=" * 80)
    
    # Replace with your actual file path
    file_path = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-160m-deduped/philpapers_train_metrics.json"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return None
    
    # Analyze with all metrics and show histograms
    result = analyze_single_file(file_path, summary_only=False)
    return result

def example_2_specific_metrics():
    """Example 2: Analyze only specific metrics."""
    print("=" * 80)
    print("EXAMPLE 2: Specific metrics analysis")
    print("=" * 80)
    
    file_path = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-160m-deduped/philpapers_train_metrics.json"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return None
    
    # Only analyze these specific metrics
    metrics_to_analyze = ['ppl', 'zlib_ratio', 'k_min_probs_0.1', 'k_max_probs_0.1']
    
    result = analyze_single_file(
        file_path, 
        metrics_to_analyze=metrics_to_analyze, 
        summary_only=True
    )
    return result

def example_3_summary_only():
    """Example 3: Quick summary analysis."""
    print("=" * 80)
    print("EXAMPLE 3: Summary-only analysis")
    print("=" * 80)
    
    file_path = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-160m-deduped/philpapers_train_metrics.json"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return None
    
    # Quick summary without histograms
    result = analyze_single_file(file_path, summary_only=True)
    return result

def example_4_all_files_in_directory():
    """Example 4: Analyze all files in a directory."""
    print("=" * 80)
    print("EXAMPLE 4: All files analysis")
    print("=" * 80)
    
    results_dir = "/scratch/ar7789/llm_dataset_inference/results"
    
    # Analyze all files with summary only
    all_results = analyze_all_files_in_directory(results_dir, summary_only=True)
    
    if all_results:
        print(f"\nAnalyzed {len(all_results)} files successfully")
        
        # Show summary of what was found
        all_metrics = set()
        for result in all_results:
            all_metrics.update(result['available_metrics'])
        
        print(f"Total unique metrics across all files: {len(all_metrics)}")
        print(f"Metrics: {sorted(all_metrics)}")
    
    return all_results

def example_5_compare_metrics():
    """Example 5: Compare specific metrics across all files."""
    print("=" * 80)
    print("EXAMPLE 5: Compare metrics across files")
    print("=" * 80)
    
    results_dir = "/scratch/ar7789/llm_dataset_inference/results"
    
    # First analyze all files
    all_results = analyze_all_files_in_directory(results_dir, summary_only=True)
    
    if not all_results:
        print("No results to compare")
        return
    
    # Compare specific metrics
    metrics_to_compare = ['ppl', 'zlib_ratio']
    
    for metric in metrics_to_compare:
        compare_metrics_across_files(all_results, metric)

def example_6_programmatic_usage():
    """Example 6: Programmatic usage - extract specific data."""
    print("=" * 80)
    print("EXAMPLE 6: Programmatic usage")
    print("=" * 80)
    
    results_dir = "/scratch/ar7789/llm_dataset_inference/results"
    results_path = Path(results_dir)
    
    # Find all JSON files
    json_files = list(results_path.rglob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Analyze each file and extract specific information
    analysis_data = []
    
    for json_file in json_files[:5]:  # Limit to first 5 files for demo
        print(f"\nAnalyzing: {json_file.name}")
        
        result = analyze_single_file(str(json_file), summary_only=True)
        
        if result:
            # Extract specific metrics
            file_data = {
                'file': str(json_file),
                'dataset': result['dataset_name'],
                'metrics': {}
            }
            
            # Extract PPL if available
            if 'ppl' in result['results']:
                ppl_stats = result['results']['ppl']
                file_data['metrics']['ppl'] = {
                    'mean': ppl_stats['mean'],
                    'std': ppl_stats['std_dev'],
                    'count': ppl_stats['count']
                }
            
            # Extract zlib_ratio if available
            if 'zlib_ratio' in result['results']:
                zlib_stats = result['results']['zlib_ratio']
                file_data['metrics']['zlib_ratio'] = {
                    'mean': zlib_stats['mean'],
                    'std': zlib_stats['std_dev'],
                    'count': zlib_stats['count']
                }
            
            analysis_data.append(file_data)
            
            # Print summary
            if 'ppl' in file_data['metrics']:
                ppl = file_data['metrics']['ppl']
                print(f"  PPL: mean={ppl['mean']:.3f}, std={ppl['std']:.3f}, count={ppl['count']}")
            
            if 'zlib_ratio' in file_data['metrics']:
                zlib = file_data['metrics']['zlib_ratio']
                print(f"  Zlib: mean={zlib['mean']:.3f}, std={zlib['std']:.3f}, count={zlib['count']}")
    
    return analysis_data

def main():
    """Run all examples."""
    print("LLM Dataset Inference Analysis Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_1_single_file_analysis()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_specific_metrics()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_summary_only()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_all_files_in_directory()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_compare_metrics()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_programmatic_usage()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")

if __name__ == "__main__":
    main()


