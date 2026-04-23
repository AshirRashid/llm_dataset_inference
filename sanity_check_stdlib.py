import os
import csv
import math
import random

def manual_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1: return float('nan')
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2
    var1 = sum((x - mean1)**2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2)**2 for x in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0: return 0.0
    return (mean1 - mean2) / pooled_std

print("--- Unit Testing effect_size.py Logic ---")
random.seed(42)
group1 = [random.gauss(0.0, 1.0) for _ in range(1000)]
group2 = [random.gauss(0.5, 1.0) for _ in range(1000)]
d_actual = manual_cohens_d(group1, group2)
print(f"Synthetic TP test (Expected ~ -0.5): {d_actual:.4f}")

group1_fp = [random.gauss(0.0, 1.0) for _ in range(1000)]
group2_fp = [random.gauss(0.0, 1.0) for _ in range(1000)]
d_actual_fp = manual_cohens_d(group1_fp, group2_fp)
print(f"Synthetic FP test (Expected ~ 0.0): {d_actual_fp:.4f}\n")

print("--- Aggregated Results Sanity Check ---")
base_dir = "/scratch/ar7789/llm_dataset_inference/aggregated_results/cohens_d/mean+p-value-outliers"
tp_dir = os.path.join(base_dir, "train-normalize", "EleutherAI_pythia-12b-deduped")
fp_dir = os.path.join(base_dir, "train-normalize-1-false_positive", "EleutherAI_pythia-12b-deduped")

def analyze_dir(directory, condition):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return
    results = []
    print(f"[{condition}]")
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            path = os.path.join(directory, file)
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                all_vals = []
                for row in reader:
                    # skip seed
                    vals = [float(x) for x in row[1:]]
                    all_vals.extend(vals)
                if len(all_vals) > 0:
                    mean_val = sum(all_vals) / len(all_vals)
                    var_val = sum((x - mean_val)**2 for x in all_vals) / len(all_vals)
                    std_val = math.sqrt(var_val)
                    print(f"  {file}: Actual Mean Cohen's D = {mean_val:.4f} \u00b1 {std_val:.4f}")
            
analyze_dir(tp_dir, "TP (True Positive - Expected != 0)")
print("")
analyze_dir(fp_dir, "FP (False Positive - Expected ~ 0)")

