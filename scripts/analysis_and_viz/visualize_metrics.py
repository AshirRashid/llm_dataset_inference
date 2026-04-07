#!/usr/bin/env python3
"""
Simple visualization script for LLM dataset inference metrics.
This script loads JSON metrics files and creates basic visualizations.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
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

def create_histogram(data, metric_name, dataset_name, output_dir):
    """Create a histogram for a specific metric."""
    if metric_name not in data:
        print(f"Metric '{metric_name}' not found in data")
        return
    
    values = data[metric_name]
    if not isinstance(values, list):
        print(f"Metric '{metric_name}' is not a list")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {metric_name} - {dataset_name}')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.3f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = os.path.join(output_dir, f'{dataset_name}_{metric_name}_histogram.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram: {output_file}")

def create_boxplot(all_data, metric_name, output_dir):
    """Create a box plot comparing different datasets for a metric."""
    datasets = []
    labels = []
    
    for dataset_name, data in all_data.items():
        if metric_name in data and isinstance(data[metric_name], list):
            datasets.append(data[metric_name])
            labels.append(dataset_name)
    
    if not datasets:
        print(f"No valid data found for metric '{metric_name}'")
        return
    
    plt.figure(figsize=(12, 8))
    plt.boxplot(datasets, labels=labels)
    plt.title(f'Box Plot Comparison: {metric_name}')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join(output_dir, f'{metric_name}_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved box plot: {output_file}")

def create_lineplot(data, metric_name, dataset_name, output_dir):
    """Create a line plot for a metric (useful for time series data)."""
    if metric_name not in data:
        print(f"Metric '{metric_name}' not found in data")
        return
    
    values = data[metric_name]
    if not isinstance(values, list):
        print(f"Metric '{metric_name}' is not a list")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(values, alpha=0.7)
    plt.title(f'{metric_name} over samples - {dataset_name}')
    plt.xlabel('Sample Index')
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(values) > 10:
        window_size = min(100, len(values) // 10)
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(values)), moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    output_file = os.path.join(output_dir, f'{dataset_name}_{metric_name}_lineplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved line plot: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize LLM dataset inference metrics')
    parser.add_argument('--results_dir', default='results', help='Path to results directory')
    parser.add_argument('--output_dir', default='visualizations', help='Output directory for plots')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to visualize (default: all)')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to visualize (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSON files
    results_path = Path(args.results_dir)
    json_files = list(results_path.rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {args.results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Load all data
    all_data = {}
    for json_file in json_files:
        # Extract dataset name from path
        relative_path = json_file.relative_to(results_path)
        dataset_name = f"{relative_path.parent.name}_{relative_path.stem}"
        
        data = load_metrics_file(json_file)
        if data:
            all_data[dataset_name] = data
            print(f"Loaded: {dataset_name}")
    
    if not all_data:
        print("No valid data loaded")
        return
    
    # Get available metrics
    all_metrics = set()
    for data in all_data.values():
        all_metrics.update(get_available_metrics(data))
    
    print(f"Available metrics: {sorted(all_metrics)}")
    
    # Filter metrics and datasets if specified
    metrics_to_plot = args.metrics if args.metrics else sorted(all_metrics)
    datasets_to_plot = args.datasets if args.datasets else list(all_data.keys())
    
    # Filter data
    filtered_data = {k: v for k, v in all_data.items() if k in datasets_to_plot}
    
    print(f"Creating visualizations for {len(metrics_to_plot)} metrics across {len(filtered_data)} datasets")
    
    # Create visualizations
    for metric in metrics_to_plot:
        if metric not in all_metrics:
            print(f"Warning: Metric '{metric}' not found in data")
            continue
        
        print(f"\nProcessing metric: {metric}")
        
        # Create box plot comparing all datasets
        create_boxplot(filtered_data, metric, args.output_dir)
        
        # Create individual plots for each dataset
        for dataset_name, data in filtered_data.items():
            if metric in data:
                create_histogram(data, metric, dataset_name, args.output_dir)
                create_lineplot(data, metric, dataset_name, args.output_dir)
    
    print(f"\nVisualization complete! Check the '{args.output_dir}' directory for output files.")

if __name__ == "__main__":
    main()

