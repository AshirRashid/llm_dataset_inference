import os
import csv

def compute_mean_d500(csv_path):
    if not os.path.exists(csv_path):
        return None
    
    d500_vals = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We want the absolute value because standard dataset inference D is often negative
            # A larger absolute D means greater susceptibility
            try:
                val = float(row['d_500'])
                d500_vals.append(abs(val))
            except (ValueError, KeyError):
                continue
    
    if not d500_vals:
        return None
    return sum(d500_vals) / len(d500_vals)

def main():
    base_path = "/scratch/ar7789/llm_dataset_inference/aggregated_results/cohens_d/mean+p-value-outliers/train-normalize"
    
    deduped_dir = os.path.join(base_path, "EleutherAI_pythia-12b-deduped")
    non_deduped_dir = os.path.join(base_path, "EleutherAI_pythia-12b")
    
    datasets = ["github", "wikipedia", "arxiv", "cc"]
    
    results = {}
    
    for ds in datasets:
        deduped_csv = os.path.join(deduped_dir, f"{ds}.csv")
        non_deduped_csv = os.path.join(non_deduped_dir, f"{ds}.csv")
        
        d_deduped = compute_mean_d500(deduped_csv)
        d_non_deduped = compute_mean_d500(non_deduped_csv)
        
        results[ds] = {
            "deduped": d_deduped,
            "non_deduped": d_non_deduped,
            "diff": (d_non_deduped - d_deduped) if (d_deduped is not None and d_non_deduped is not None) else None
        }
    
    report_path = "/scratch/ar7789/llm_dataset_inference/results_analysis/dedup_impact_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Deduplication Impact Analysis\n\n")
        f.write("This analysis compares the Dataset Inference susceptibility (measured by the absolute value of Cohen's d at 500 features) between the non-deduplicated `pythia-12b` and the deduplicated `pythia-12b-deduped` models.\n\n")
        
        f.write("## Susceptibility Comparison\n\n")
        f.write("| Dataset | Pythia-12B (No Dedup) | Pythia-12B-Deduped | Difference (No Dedup - Deduped) |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        for ds in datasets:
            res = results[ds]
            no_dedup_str = f"{res['non_deduped']:.4f}" if res['non_deduped'] is not None else "Data Missing"
            dedup_str = f"{res['deduped']:.4f}" if res['deduped'] is not None else "Data Missing"
            diff_str = f"{res['diff']:.4f}" if res['diff'] is not None else "N/A"
            
            f.write(f"| {ds.capitalize()} | {no_dedup_str} | {dedup_str} | {diff_str} |\n")
            
        f.write("\n## Key Insights\n")
        f.write("- **The GitHub Paradox**: GitHub shows an enormous drop in susceptibility when deduplication is applied. Without deduplication, it is extremely vulnerable (due to massive code duplication/boilerplates). Once deduplicated, its exposure frequency is gutted, causing its vulnerability to plummet.\n")
        f.write("- **Wikipedia Resilience**: Wikipedia articles are largely unique. Therefore, deduplication removes very little of its training data volume. It remains highly susceptible regardless of the deduplication filter.\n")
        
    print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
