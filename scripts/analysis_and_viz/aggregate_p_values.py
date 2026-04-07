import os
import csv
import sys
import traceback

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Configuration Constants ---
FILTER_STRINGS = ["pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped",  "pythia-12b-deduped"]
DEFAULT_START_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/mean+p-value-outliers"
DEFAULT_OUTPUT_CSV = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/agg_plots/{filter_string}-{dataset_name}.csv"
THRESHOLD_VALUE = 0.1
FP_THRESHOLD_VALUE = 0.5


def get_dataset_files(start_dir):
    """
    Scans the start_dir to find the first subdirectory containing CSV files.
    Returns a list of CSV filenames found (e.g., ['github.csv', 'arxiv.csv']).
    """
    for root, dirs, files in os.walk(start_dir):
        # We are looking for the leaf directories that contain the CSVs.
        # A simple heuristic is to check if any .csv file exists in valid subdirs.
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            return csv_files
    return []

def aggregate_p_values(start_dir, output_csv, csv_filename, filter_string=None):
    """
    Traverses the start_dir to find specific csv_filename files.
    Extracts p-values for ALL seeds.
    Writes a long-format CSV: model_subdir, run_type, seed, p_500, ...
    """
    
    print(f"Searching for '{csv_filename}' files in: {start_dir}")
    if filter_string:
        print(f"Filtering for models containing: '{filter_string}'")

    all_rows = []
    p_keys = set()
    found_count = 0

    # Traverse directory
    for root, dirs, files in os.walk(start_dir):
        if csv_filename in files:
            file_path = os.path.join(root, csv_filename)
            model_subdir_name = os.path.basename(root)

            # Apply filter
            if filter_string and filter_string not in model_subdir_name:
                continue
            
            # Check if this is a false positive run
            is_false_positive = "false_positive" in root
            run_type = "false_positive" if is_false_positive else "regular"

            if is_false_positive:
                print(f"Found false positive file: {file_path}")

            try:
                with open(file_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames if reader.fieldnames else []
                    
                    # Identify p_value keys dynamically
                    current_p_keys = [k for k in fieldnames if k.startswith('p_')]
                    for k in current_p_keys:
                        p_keys.add(k)
                    
                    # Read ALL data rows (seeds)
                    for data_row in reader:
                        row_data = {
                            'model_subdir': model_subdir_name,
                            'run_type': run_type,
                            'seed': data_row.get('seed', 'unknown'),
                        }
                        for k in current_p_keys:
                            row_data[k] = data_row.get(k)
                        
                        all_rows.append(row_data)
                        found_count += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Sort p_keys
    sorted_p_keys = list(p_keys)
    sorted_p_keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
    
    header = ['model_subdir', 'run_type', 'seed'] + sorted_p_keys

    # Sort rows by model name, then run_type, then seed
    all_rows.sort(key=lambda x: (x['model_subdir'], x['run_type'], x['seed']))

    print(f"Found {found_count} matching entries. Writing results to: {output_csv}")
    
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Aggregation complete.")
    except Exception as e:
        print(f"Error writing output file: {e}")

def get_model_label(model_name, filter_string):
    """
    Returns a human-readable label for a model subdirectory.
    """
    if not filter_string or filter_string not in model_name:
        return model_name
    
    parts = model_name.split(filter_string)
    suffix = parts[-1] if len(parts) > 1 else ""
    return suffix if suffix else "Original"

def model_sort_key(model_name, filter_string):
    """
    Sort key to group by family and order by bits, with 'Original' last.
    """
    if not filter_string or filter_string not in model_name:
        return (model_name, 999)

    suffix = model_name.split(filter_string)[-1]
    if not suffix:
        return (filter_string, 999) # Original last

    import re
    # Look for bits like -b2, -b3, -b4, -b8
    bit_match = re.search(r'-b(\d+)', suffix)
    if bit_match:
        return (filter_string, int(bit_match.group(1)))
    
    return (filter_string, 500) # Other variations in middle

def visualize_results(csv_path, dataset_name, filter_string=None):
    """
    Visualizes the aggregated p-values CSV.
    Box plots p_500 for Regular (Blue) and False Positive (Red) runs, aggregating across seeds.
    Truncates x-axis labels if filter_string is provided.
    """

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("CSV is empty. Nothing to visualize.")
            return

        if 'p_500' not in df.columns:
            print("Column 'p_500' not found. Cannot plot.")
            return

        # Ensure numeric p_500
        df['p_500'] = pd.to_numeric(df['p_500'], errors='coerce')

        # Get unique models and sort them using our custom key
        unique_models = sorted(df['model_subdir'].unique(), key=lambda x: model_sort_key(x, filter_string))
        
        # Create mapping for positions
        model_to_idx = {model: i for i, model in enumerate(unique_models)}
        df['model_idx'] = df['model_subdir'].map(model_to_idx)
        
        # Generate display labels
        display_labels = [get_model_label(m, filter_string) for m in unique_models]

        plt.figure(figsize=(15, 8))

        # prepare data for boxplot
        # We want to group by model_code and run_type
        # Position: model_code * 2 + (0 for reg, 1 for fp) ?
        # Or simpler: just use positions directly
        
        positions_reg = []
        data_reg = []
        positions_fp = []
        data_fp = []

        width = 0.35

        for idx, model_name in enumerate(unique_models):
            # Regular
            reg_data = df[(df['model_idx'] == idx) & (df['run_type'] == 'regular')]['p_500'].dropna().tolist()
            if reg_data:
                positions_reg.append(idx - width/2)
                data_reg.append(reg_data)
            
            # FP
            fp_data = df[(df['model_idx'] == idx) & (df['run_type'] == 'false_positive')]['p_500'].dropna().tolist()
            if fp_data:
                positions_fp.append(idx + width/2)
                data_fp.append(fp_data)

        # Plot Boxplots
        if data_reg:
            bp_reg = plt.boxplot(data_reg, positions=positions_reg, widths=width, patch_artist=True, 
                                 boxprops=dict(facecolor="lightblue", color="blue"),
                                 medianprops=dict(color="blue"),
                                 showfliers=True)
        
        if data_fp:
            bp_fp = plt.boxplot(data_fp, positions=positions_fp, widths=width, patch_artist=True,
                                boxprops=dict(facecolor="lightcoral", color="red"),
                                medianprops=dict(color="red"),
                                showfliers=True)

        # Draw thresholds
        plt.axhline(y=THRESHOLD_VALUE, color='black', linestyle=':', label=f'Threshold p={THRESHOLD_VALUE}')
        plt.axhline(y=FP_THRESHOLD_VALUE, color='black', linestyle=':', label=f'Threshold p={FP_THRESHOLD_VALUE}')

        plt.title(f'P-Values (p_500) Distribution by Model - {dataset_name}')
        plt.xlabel('Model Subdirectory')
        plt.ylabel('P-Value')
        
        # Set x-ticks
        plt.xticks(range(len(unique_models)), display_labels, rotation=90, fontsize=8)
        
        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Regular'),
            Line2D([0], [0], color='red', lw=4, label='False Positive'),
            Line2D([0], [0], color='black',linestyle=':', label='Thresholds')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        output_img = os.path.splitext(csv_path)[0] + '.png'
        plt.savefig(output_img)
        print(f"Visualization saved to: {output_img}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        traceback.print_exc()


def visualize_grid(start_dir, output_csv_template, dataset_files, filter_strings):
    """
    Creates a grid of plots:
    - Rows: Distinct models (from filter_strings)
    - Columns: Distinct splits (from dataset_files) * 2 (Regular, FP)
    """
    try:
        # Create grid: 2 columns per dataset (Regular, FP)
        nrows = len(filter_strings)
        n_datasets = len(dataset_files)
        ncols = n_datasets * 2
        
        # Adjust figsize: wider to accommodate double columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        
        # Get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']

        for row_idx, filter_string in enumerate(filter_strings):
            is_bottom_row = (row_idx == nrows - 1)
            
            for ds_idx, csv_filename in enumerate(dataset_files):
                dataset_name = os.path.splitext(csv_filename)[0]
                csv_path = output_csv_template.format(filter_string=filter_string, dataset_name=dataset_name)
                
                # Subplot indices
                col_reg = ds_idx * 2
                col_fp = ds_idx * 2 + 1
                
                ax_reg = axes[row_idx, col_reg]
                ax_fp = axes[row_idx, col_fp]
                
                # Titles (only top row)
                if row_idx == 0:
                    ax_reg.set_title(f"{dataset_name} (Regular)")
                    ax_fp.set_title(f"{dataset_name} (FP)")
                
                # Y-labels (only first column)
                if col_reg == 0:
                    ax_reg.set_ylabel(f"{filter_string}\nP-Value")
                
                # Load Data
                if not os.path.exists(csv_path):
                    for ax in [ax_reg, ax_fp]:
                        ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    if df.empty or 'p_500' not in df.columns:
                        for ax in [ax_reg, ax_fp]:
                            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                        continue

                    # Process Data
                    # Get unique models and sort them using our custom key
                    unique_models = sorted(df['model_subdir'].unique(), key=lambda x: model_sort_key(x, filter_string))
                    
                    # Create mapping for positions
                    model_to_idx = {model: i for i, model in enumerate(unique_models)}
                    df['model_idx'] = df['model_subdir'].map(model_to_idx)
                    
                    # Generate display labels
                    display_labels = [get_model_label(m, filter_string) for m in unique_models]

                    df['p_500'] = pd.to_numeric(df['p_500'], errors='coerce')
                    
                    # Prepare Data for Boxplots
                    data_reg = []
                    positions_reg = []
                    for idx, model_name in enumerate(unique_models):
                         reg_vals = df[(df['model_idx'] == idx) & (df['run_type'] == 'regular')]['p_500'].dropna().tolist()
                         if reg_vals:
                             data_reg.append(reg_vals)
                             positions_reg.append(idx)
                    
                    data_fp = []
                    positions_fp = []
                    for idx, model_name in enumerate(unique_models):
                         fp_vals = df[(df['model_idx'] == idx) & (df['run_type'] == 'false_positive')]['p_500'].dropna().tolist()
                         if fp_vals:
                             data_fp.append(fp_vals)
                             positions_fp.append(idx)

                    # Plot Regular Boxplots
                    if data_reg:
                        ax_reg.boxplot(data_reg, positions=positions_reg, widths=0.5, patch_artist=True,
                                        boxprops=dict(facecolor="lightblue", color="blue"),
                                        medianprops=dict(color="blue"), showfliers=True)
                    else:
                        ax_reg.text(0.5, 0.5, "No Regular Data", ha='center', va='center')

                    # Plot FP Boxplots
                    if data_fp:
                        ax_fp.boxplot(data_fp, positions=positions_fp, widths=0.5, patch_artist=True,
                                       boxprops=dict(facecolor="lightcoral", color="red"),
                                       medianprops=dict(color="red"), showfliers=True)
                    else:
                        ax_fp.text(0.5, 0.5, "No FP Data", ha='center', va='center')

                    # Formatting for both axes
                    for ax in [ax_reg, ax_fp]:
                        ax.axhline(y=THRESHOLD_VALUE, color='black', linestyle=':', alpha=0.5)
                        ax.axhline(y=FP_THRESHOLD_VALUE, color='black', linestyle=':', alpha=0.5)
                        ax.set_ylim(-0.05, 1.05) # Fixed margins for comparison
                        
                        # Only set X-ticks/labels if it's the bottom row
                        if is_bottom_row:
                            ax.set_xticks(range(len(unique_models)))
                            ax.set_xticklabels(display_labels, rotation=90, fontsize=6)
                        else:
                            ax.set_xticks([]) # Hide ticks for non-bottom rows

                except Exception as e:
                    print(f"Error plotting {csv_path}: {e}")
                    for ax in [ax_reg, ax_fp]:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center')

        # Global Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Regular Distribution'),
            Line2D([0], [0], color='red', lw=4, label='False Positive Distribution'),
            Line2D([0], [0], color='black',linestyle=':', label='Thresholds')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)

        plt.tight_layout()
        grid_output = os.path.join(os.path.dirname(DEFAULT_OUTPUT_CSV), "aggregated_p_values_grid.png")
        plt.savefig(grid_output, bbox_inches='tight')
        print(f"Grid visualization saved to: {grid_output}")

    except Exception as e:
        print(f"Error during grid visualization: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_START_DIR):
        print(f"Error: Directory not found: {DEFAULT_START_DIR}")
        sys.exit(1)

    csv_files = get_dataset_files(DEFAULT_START_DIR)
    if not csv_files:
        print(f"No CSV files found in subdirectories of {DEFAULT_START_DIR}")
        sys.exit(0)
    
    print(f"Found dataset CSVs: {csv_files}")

    for filter_string in FILTER_STRINGS:
        for csv_filename in csv_files:
            dataset_name = os.path.splitext(csv_filename)[0]
            output_csv = DEFAULT_OUTPUT_CSV.format(filter_string=filter_string, dataset_name=dataset_name)
            
            aggregate_p_values(DEFAULT_START_DIR, output_csv, csv_filename, filter_string)
            visualize_results(output_csv, dataset_name, filter_string)

    # Generate Grid Plot
    visualize_grid(DEFAULT_START_DIR, DEFAULT_OUTPUT_CSV, csv_files, FILTER_STRINGS)

