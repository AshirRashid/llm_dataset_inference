import csv
from collections import defaultdict
import os

def analyze_p_values():
    csv_path = "/scratch/ar7789/llm_dataset_inference/results_analysis/p_500_summary.csv"
    if not os.path.exists(csv_path):
        print("CSV not found!")
        return
        
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['Mean P_500'] = float(row['Mean P_500'])
            records.append(row)
            
    report_path = "/scratch/ar7789/llm_dataset_inference/results_analysis/analysis_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Analysis of Aggregated Results\n\n")
        f.write("## Overview\n")
        f.write(f"Total records analyzed: {len(records)}\n\n")
        
        # 1. Base Model breakdown
        f.write("## Base Model Breakdown\n")
        f.write("Average p-value by Base Model:\n\n")
        f.write("| Base Model | Mean P_500 |\n")
        f.write("| --- | --- |\n")
        base_scores = defaultdict(list)
        for r in records: base_scores[r["Base Model"]].append(r["Mean P_500"])
        for m, scores in sorted(base_scores.items()):
            f.write(f"| {m} | {sum(scores)/len(scores):.4e} |\n")
        f.write("\n")
        
        # 2. Dataset breakdown
        f.write("## Dataset Breakdown\n")
        f.write("Average p-value by Dataset (sorted from lowest to highest, lowest indicates highest privacy leakage/inference signal):\n\n")
        f.write("| Dataset | Mean P_500 |\n")
        f.write("| --- | --- |\n")
        ds_scores = defaultdict(list)
        for r in records: ds_scores[r["Dataset"]].append(r["Mean P_500"])
        ds_avgs = [(ds, sum(scores)/len(scores)) for ds, scores in ds_scores.items()]
        for ds, avg in sorted(ds_avgs, key=lambda x: x[1]):
            f.write(f"| {ds} | {avg:.4e} |\n")
        f.write("\n")
        
        # 3. Config breakdown
        f.write("## Quantization Config Breakdown\n")
        f.write("Average p-value by Quantization Configuration:\n\n")
        f.write("| Config | Mean P_500 |\n")
        f.write("| --- | --- |\n")
        cfg_scores = defaultdict(list)
        for r in records: cfg_scores[r["Config"]].append(r["Mean P_500"])
        cfg_avgs = [(cfg, sum(scores)/len(scores)) for cfg, scores in cfg_scores.items()]
        for cfg, avg in sorted(cfg_avgs, key=lambda x: x[1]):
            f.write(f"| {cfg} | {avg:.4e} |\n")
        f.write("\n")
        
        # 4. Highest Leakage combinations
        f.write("## Top 15 Highest Privacy Leakage Configurations (Lowest P-Values)\n\n")
        f.write("| Base Model | Dataset | Config | Mean P_500 |\n")
        f.write("| --- | --- | --- | --- |\n")
        sorted_records = sorted(records, key=lambda x: x["Mean P_500"])
        for r in sorted_records[:15]:
            f.write(f"| {r['Base Model']} | {r['Dataset']} | {r['Config']} | {r['Mean P_500']:.4e} |\n")
        f.write("\n")

    print(f"Analysis saved to {report_path}")

if __name__ == '__main__':
    analyze_p_values()
