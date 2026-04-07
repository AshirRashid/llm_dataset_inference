#!/usr/bin/env python3
"""
Simple visualization script for LLM dataset inference metrics.
This script loads JSON metrics files and creates basic text-based visualizations.
"""

import json
import os
import math
from pathlib import Path
import argparse

def load_metrics_file(file_path):
    """Load and parse a JSON metrics file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_available_metrics(data):
    """Extract available metric names from the data."""
    if isinstance(data, dict):
        return list(data.keys())
    return []

def calculate_stats(values):
    """Calculate basic statistics for a list of values."""
    if not values:
        return {}
    
    n = len(values)
    mean = sum(values) / n
    
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    
    # Sort for percentiles
    sorted_values = sorted(values)
    
    # Calculate percentiles
    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1
    
    return {
        'count': n,
        'mean': mean,
        'std_dev': std_dev,
        'min': min(values),
        'max': max(values),
        'median': percentile(sorted_values, 50),
        'q25': percentile(sorted_values, 25),
        'q75': percentile(sorted_values, 75)
    }

def create_text_histogram(values, bins=20, width=60):
    """Create a simple text-based histogram."""
    if not values:
        return "No data"
    
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / bins
    
    # Create bins
    histogram = [0] * bins
    for value in values:
        bin_idx = min(int((value - min_val) / bin_width), bins - 1)
        histogram[bin_idx] += 1
    
    # Find max count for scaling
    max_count = max(histogram)
    
    result = []
    for i, count in enumerate(histogram):
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        bar = '█' * bar_length
        result.append(f"{bin_start:8.3f}-{bin_end:8.3f} |{bar} {count}")
    
    return '\n'.join(result)

def print_metric_summary(data, metric_name, dataset_name):
    """Print a summary of a metric for a dataset."""
    if metric_name not in data:
        print(f"Metric '{metric_name}' not found in {dataset_name}")
        return
    
    values = data[metric_name]
    if not isinstance(values, list):
        print(f"Metric '{metric_name}' is not a list in {dataset_name}")
        return
    
    stats = calculate_stats(values)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric_name}")
    print(f"{'='*60}")
    print(f"Count:     {stats['count']:,}")
    print(f"Mean:      {stats['mean']:.6f}")
    print(f"Std Dev:   {stats['std_dev']:.6f}")
    print(f"Min:       {stats['min']:.6f}")
    print(f"Max:       {stats['max']:.6f}")
    print(f"Median:    {stats['median']:.6f}")
    print(f"Q25:       {stats['q25']:.6f}")
    print(f"Q75:       {stats['q75']:.6f}")
    
    print(f"\nHistogram:")
    print(create_text_histogram(values))

def compare_metrics(all_data, metric_name):
    """Compare a metric across all datasets."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {metric_name}")
    print(f"{'='*80}")
    
    comparison_data = []
    for dataset_name, data in all_data.items():
        if metric_name in data and isinstance(data[metric_name], list):
            stats = calculate_stats(data[metric_name])
            comparison_data.append((dataset_name, stats))
    
    if not comparison_data:
        print(f"No data found for metric '{metric_name}'")
        return
    
    # Sort by mean value
    comparison_data.sort(key=lambda x: x[1]['mean'])
    
    print(f"{'Dataset':<40} {'Count':<8} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print('-' * 100)
    
    for dataset_name, stats in comparison_data:
        print(f"{dataset_name:<40} {stats['count']:<8,} {stats['mean']:<12.6f} {stats['std_dev']:<12.6f} {stats['min']:<12.6f} {stats['max']:<12.6f}")

def analyze_single_file(file_path, metrics_to_analyze=None, summary_only=False):
    """
    Analyze a single metrics file.
    
    Args:
        file_path (str): Absolute path to the JSON metrics file
        metrics_to_analyze (list, optional): List of specific metrics to analyze. If None, analyzes all.
        summary_only (bool): If True, show only summary statistics, no detailed breakdowns.
    
    Returns:
        dict: Dictionary containing the loaded data and analysis results
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return None
    
    if not file_path.suffix == '.json':
        print(f"Error: File {file_path} is not a JSON file")
        return None
    
    # Load data
    data = load_metrics_file(file_path)
    if not data:
        return None
    
    # Extract dataset name from path
    dataset_name = f"{file_path.parent.name}_{file_path.stem}"
    
    print(f"\n{'='*80}")
    print(f"ANALYZING FILE: {file_path}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Get available metrics
    available_metrics = get_available_metrics(data)
    print(f"Available metrics: {sorted(available_metrics)}")
    
    # Filter metrics if specified
    metrics_to_plot = metrics_to_analyze if metrics_to_analyze else sorted(available_metrics)
    
    # Filter to only include metrics that exist in the data
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
    
    if not metrics_to_plot:
        print("No valid metrics to analyze")
        return None
    
    print(f"\nAnalyzing {len(metrics_to_plot)} metrics")
    
    # Analyze each metric
    results = {}
    for metric in metrics_to_plot:
        if metric not in data:
            print(f"Warning: Metric '{metric}' not found in data")
            continue
        
        values = data[metric]
        if not isinstance(values, list):
            print(f"Warning: Metric '{metric}' is not a list")
            continue
        
        stats = calculate_stats(values)
        results[metric] = stats
        
        print(f"\n{'-'*60}")
        print(f"Metric: {metric}")
        print(f"{'-'*60}")
        print(f"Count:     {stats['count']:,}")
        print(f"Mean:      {stats['mean']:.6f}")
        print(f"Std Dev:   {stats['std_dev']:.6f}")
        print(f"Min:       {stats['min']:.6f}")
        print(f"Max:       {stats['max']:.6f}")
        print(f"Median:    {stats['median']:.6f}")
        print(f"Q25:       {stats['q25']:.6f}")
        print(f"Q75:       {stats['q75']:.6f}")
        
        if not summary_only:
            print(f"\nHistogram:")
            print(create_text_histogram(values))
    
    return {
        'file_path': str(file_path),
        'dataset_name': dataset_name,
        'data': data,
        'results': results,
        'available_metrics': available_metrics
    }

def analyze_all_files_in_directory(results_dir, metrics_to_analyze=None, summary_only=False):
    """
    Analyze all JSON files in a directory.
    
    Args:
        results_dir (str): Path to results directory
        metrics_to_analyze (list, optional): List of specific metrics to analyze. If None, analyzes all.
        summary_only (bool): If True, show only summary statistics, no detailed breakdowns.
    
    Returns:
        list: List of analysis results for each file
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return []
    
    # Find all JSON files
    json_files = list(results_path.rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return []
    
    print(f"Found {len(json_files)} JSON files")
    
    # Analyze each file
    all_results = []
    for json_file in json_files:
        result = analyze_single_file(str(json_file), metrics_to_analyze, summary_only)
        if result:
            all_results.append(result)
    
    return all_results

def compare_metrics_across_files(all_results, metric_name):
    """Compare a metric across all analyzed files."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {metric_name}")
    print(f"{'='*80}")
    
    comparison_data = []
    for result in all_results:
        if metric_name in result['results']:
            stats = result['results'][metric_name]
            comparison_data.append((result['dataset_name'], stats))
    
    if not comparison_data:
        print(f"No data found for metric '{metric_name}'")
        return
    
    # Sort by mean value
    comparison_data.sort(key=lambda x: x[1]['mean'])
    
    print(f"{'Dataset':<50} {'Count':<8} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print('-' * 110)
    
    for dataset_name, stats in comparison_data:
        print(f"{dataset_name:<50} {stats['count']:<8,} {stats['mean']:<12.6f} {stats['std_dev']:<12.6f} {stats['min']:<12.6f} {stats['max']:<12.6f}")

def main():
    parser = argparse.ArgumentParser(description='Simple text-based visualization for LLM dataset inference metrics')
    parser.add_argument('--file', help='Absolute path to a single JSON file to analyze')
    parser.add_argument('--results_dir', default='results', help='Path to results directory')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to visualize (default: all)')
    parser.add_argument('--summary_only', action='store_true', help='Show only summary statistics, no detailed breakdowns')
    parser.add_argument('--compare', action='store_true', help='Compare metrics across all files')
    
    args = parser.parse_args()
    
    if args.file:
        # Analyze single file
        result = analyze_single_file(args.file, args.metrics, args.summary_only)
        if not result:
            return
    else:
        # Analyze all files in directory
        all_results = analyze_all_files_in_directory(args.results_dir, args.metrics, args.summary_only)
        
        if not all_results:
            print("No valid data loaded")
            return
        
        if args.compare:
            # Get all available metrics across all files
            all_metrics = set()
            for result in all_results:
                all_metrics.update(result['available_metrics'])
            
            metrics_to_compare = args.metrics if args.metrics else sorted(all_metrics)
            
            for metric in metrics_to_compare:
                if metric in all_metrics:
                    compare_metrics_across_files(all_results, metric)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()

