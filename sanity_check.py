import os
import json
import numpy as np
import pandas as pd
from effect_size import cohens_d
import scipy.stats

# 1. Sanity check of effect_size.py itself
print("--- Unit Testing effect_size.py ---")
np.random.seed(42)
group1 = np.random.normal(loc=0.0, scale=1.0, size=1000)
group2 = np.random.normal(loc=0.5, scale=1.0, size=1000)

d_actual = cohens_d(group1, group2)
# Expected is roughly (0.0 - 0.5) / 1.0 = -0.5
print(f"Synthetic TP test (Expected ~ -0.5): {d_actual:.4f}")

group1_fp = np.random.normal(loc=0.0, scale=1.0, size=1000)
group2_fp = np.random.normal(loc=0.0, scale=1.0, size=1000)
d_actual_fp = cohens_d(group1_fp, group2_fp)
print(f"Synthetic FP test (Expected ~ 0.0): {d_actual_fp:.4f}\n")

# 2. Analyze the actual generated values for TP vs FP
print("--- Aggregated Results Sanity Check ---")
base_dir = "/scratch/ar7789/llm_dataset_inference/aggregated_results/cohens_d/mean+p-value-outliers"
tp_dir = os.path.join(base_dir, "train-normalize", "EleutherAI_pythia-12b-deduped")
fp_dir = os.path.join(base_dir, "train-normalize-1-false_positive", "EleutherAI_pythia-12b-deduped")

def analyze_dir(directory, condition):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            path = os.path.join(directory, file)
            df = pd.read_csv(path)
            # mean across all columns except seed
            numeric_cols = [c for c in df.columns if c != 'seed']
            mean_vals = df[numeric_cols].mean(axis=1).mean()
            std_vals = df[numeric_cols].mean(axis=1).std()
            print(f"[{condition}] {file}: Actual Mean Cohen's D = {mean_vals:.4f} \u00b1 {std_vals:.4f}")

analyze_dir(tp_dir, "TP (True Positive - Expected != 0)")
print("")
analyze_dir(fp_dir, "FP (False Positive - Expected ~ 0)")
