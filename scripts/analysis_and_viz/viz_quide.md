# LLM Dataset Inference Visualization

Simple Python scripts to visualize metrics from LLM dataset inference results.

## Scripts

### 1. `visualize_metrics.py` - Full Visualization (requires matplotlib/numpy)
Creates three types of visualizations:
1. **Histograms** - Show the distribution of each metric across samples
2. **Box Plots** - Compare metrics across different datasets  
3. **Line Plots** - Show metric values over sample indices (with moving average)

### 2. `simple_visualize.py` - Text-based Analysis (no dependencies)
Creates text-based visualizations:
1. **Summary Statistics** - Mean, std dev, min, max, median, quartiles
2. **Text Histograms** - ASCII bar charts showing distributions
3. **Cross-dataset Comparisons** - Tabular comparisons of metrics

## Installation

### For Full Visualization (with plots)
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### For Text-based Analysis
No additional dependencies required - uses only Python standard library.

## Usage

### Text-based Analysis (Recommended - No Dependencies)

Quick summary of all metrics across datasets:

```bash
python simple_visualize.py --summary_only
```

Detailed analysis with histograms:

```bash
python simple_visualize.py
```

Analyze specific metrics or datasets:

```bash
python simple_visualize.py --metrics ppl ppl_ratio_whitespace_perturbation
python simple_visualize.py --datasets pythia-160m-deduped_arxiv_train_metrics
```

### Full Visualization (Requires matplotlib/numpy)

Run the script with default settings to visualize all metrics and datasets:

```bash
python visualize_metrics.py
```

### Advanced Usage

Specify custom directories and filter specific metrics or datasets:

```bash
# Use custom results directory
python visualize_metrics.py --results_dir /path/to/results

# Specify output directory
python visualize_metrics.py --output_dir /path/to/output

# Visualize specific metrics only
python visualize_metrics.py --metrics ppl_ratio_whitespace_perturbation ppl_diff_underscore_trick

# Visualize specific datasets only
python visualize_metrics.py --datasets pythia-160m-deduped_arxiv_train pythia-160m-deduped_github_train

# Combine options
python visualize_metrics.py --results_dir results --output_dir plots --metrics ppl_ratio_whitespace_perturbation
```

### Command Line Options

- `--results_dir`: Path to the results directory containing JSON files (default: `results`)
- `--output_dir`: Directory to save visualization plots (default: `visualizations`)
- `--metrics`: List of specific metrics to visualize (default: all available metrics)
- `--datasets`: List of specific datasets to visualize (default: all available datasets)

## Output

The script generates PNG files in the output directory:

- `{dataset}_{metric}_histogram.png` - Distribution histograms
- `{dataset}_{metric}_lineplot.png` - Time series plots
- `{metric}_boxplot.png` - Cross-dataset comparisons

## Data Format

The script expects JSON files with metric arrays, for example:

```json
{
  "ppl_ratio_whitespace_perturbation": [0.48, 0.53, 0.55, ...],
  "ppl_diff_underscore_trick": [-0.69, -0.56, -0.39, ...]
}
```

## Example

After running the script on your results, you'll get visualizations like:

- Histograms showing the distribution of perplexity ratios
- Box plots comparing different datasets
- Line plots showing how metrics change across samples

The script automatically detects available metrics and datasets from your JSON files.