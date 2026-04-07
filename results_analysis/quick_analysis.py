import glob
import os
import csv

CSV_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/mean+p-value-outliers/train-normalize"

target_models = ['12b', '1b', '410m', '160m']
records = []
files = glob.glob(os.path.join(CSV_DIR, "*", "*.csv"))

for filepath in files[:10000]: # limit to prevent hangs
    subdir = os.path.basename(os.path.dirname(filepath))
    model_size = next((m for m in target_models if f"pythia-{m}" in subdir), None)
    if not model_size: continue
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        if 'p_500' not in reader.fieldnames: continue
        p_list = []
        for row in reader:
            try:
                p_num = float(row.get('p_500', 'nan'))
                if p_num == p_num: p_list.append(p_num)
            except: pass
        if p_list:
            mean = sum(p_list) / len(p_list)
            records.append({'model_size': model_size, 'subdir': subdir, 'mean': mean})

from collections import defaultdict
print(f"Loaded {len(records)} records")

# Breakdown by model_size
size_scores = defaultdict(list)
for r in records: size_scores[r['model_size']].append(r['mean'])
for s, v in size_scores.items():
    print(f"Size: {s}, Average P_500: {sum(v)/len(v):.4e}")

# Compare Pythia-1b unquantized vs 12b quantized to 4bit
one_b_unq = [r['mean'] for r in records if r['model_size'] == '1b' and 'gptq' not in r['subdir'] and 'awq' not in r['subdir'] and 'static' not in r['subdir']]
twelve_b_4bit = [r['mean'] for r in records if r['model_size'] == '12b' and ('4bit' in r['subdir'] or 'awq-b4' in r['subdir'] or 'gptq-b4' in r['subdir'])]

print(f"1B Unquantized Avg: {sum(one_b_unq)/len(one_b_unq):.4e}" if one_b_unq else "1B Unquantized: None")
print(f"12B 4-bit Avg: {sum(twelve_b_4bit)/len(twelve_b_4bit):.4e}" if twelve_b_4bit else "12B 4-bit: None")
