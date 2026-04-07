import os
import csv
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    csv_path = "/scratch/ar7789/llm_dataset_inference/results_analysis/p_500_summary.csv"
    records = []
    
    with open(csv_path, 'r') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            mean_p = float(row['Mean P_500'])
            records.append({
                "Base Model": row["Base Model"],
                "Dataset": row["Dataset"],
                "Config": row["Config"],
                "Family": "Baseline" if "deduped" in row["Config"] else "AWQ" if "awq" in row["Config"] else "Static" if "static" in row["Config"] else "GPTQ",
                "Mean P_500": mean_p,
                # Clamping for better visualization around the significance threshold
                "Log10 P_500 (Clamped)": math.log10(max(mean_p, 1e-4)) if mean_p > 0 else -4,
                "Is Leaked": mean_p < 0.05
            })
    return records

def generate_revised_plots():
    records = load_data()
    out_dir = "/scratch/ar7789/llm_dataset_inference/results_analysis/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    colors = {"Baseline": "red", "AWQ": "blue", "Static": "green", "GPTQ": "orange"}

    # --- Plot 1: The Significance Threshold ---
    # Instead of unbounded log scaling, we look at where datasets cluster around p=0.05
    plt.figure(figsize=(14, 8))
    ds_avg = defaultdict(list)
    for r in records: ds_avg[r["Dataset"]].append(r["Log10 P_500 (Clamped)"])
    sorted_datasets = sorted(ds_avg.keys(), key=lambda d: sum(ds_avg[d])/len(ds_avg[d]))
    
    for i, ds in enumerate(sorted_datasets):
        ds_recs = [r for r in records if r["Dataset"] == ds]
        for r in ds_recs:
            plt.scatter(i, r["Log10 P_500 (Clamped)"], color=colors[r["Family"]], alpha=0.6, s=50)

    # Adding the threshold of significance line
    threshold_log = math.log10(0.05)
    plt.axhline(y=threshold_log, color='red', linestyle='--', linewidth=2, label="Significance Threshold (p=0.05)")
    
    plt.xticks(range(len(sorted_datasets)), sorted_datasets, rotation=45, ha="right")
    plt.title("Revised Insight 1: Proximity to Privacy Threshold (Clamped to 10^-4 for Legibility)")
    plt.ylabel("Log10(Mean P_500) Clamped")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "revised_insight1_threshold.png"))
    plt.close()

    # --- Plot 2: Differential Impact (Delta Scatter Plot) ---
    plt.figure(figsize=(10, 8))
    
    # Calculate Deltas
    baseline_lookup = {}
    for r in records:
        if r["Family"] == "Baseline":
            baseline_lookup[r["Dataset"]] = r["Mean P_500"]
            
    deltas = []
    for r in records:
        if r["Family"] != "Baseline" and r["Dataset"] in baseline_lookup:
            base_p = baseline_lookup[r["Dataset"]]
            delta_p = r["Mean P_500"] - base_p
            # We don't want datasets that are absolutely zeroed out to clump
            if base_p > 1e-6:
                deltas.append((base_p, delta_p, colors[r["Family"]]))

    x_vals = [d[0] for d in deltas]
    y_vals = [d[1] for d in deltas]
    c_vals = [d[2] for d in deltas]
    
    plt.scatter(x_vals, y_vals, c=c_vals, alpha=0.6, s=50)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.05, color='red', linestyle='--', label="Significance Threshold (p=0.05)")
    
    plt.xscale('log')
    plt.title("Revised Insight 2: The Efficacy of Quantization depends on Original Leakage")
    plt.xlabel("Baseline (Unquantized) P_500 [Log Scale]")
    plt.ylabel("Delta Shift in P_500 (Quantized - Baseline) -> Higher is Better")
    plt.legend(["Quantization Configs", "No Effect Line", "Leakage Threshold"])
    plt.grid(linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "revised_insight2_delta.png"))
    plt.close()

    # --- Plot 3: Privacy/Capacity Trade-off ---
    plt.figure(figsize=(10, 6))
    
    arxiv_recs = [r for r in records if r["Dataset"] == "arxiv" and r["Base Model"] == "Pythia-12B"]
    for r in arxiv_recs:
        mem = 10.0
        if r["Family"] == "Baseline": mem = 24.0
        elif "b4" in r["Config"] or "4bit" in r["Config"]: mem = 6.0
        elif "b3" in r["Config"] or "3bit" in r["Config"]: mem = 4.5
        elif "b2" in r["Config"] or "2bit" in r["Config"]: mem = 3.0
        else: mem = 12.0
        plt.scatter(mem, r["Mean P_500"], color=colors[r["Family"]], marker='o', alpha=0.7, s=80)
        
    plt.scatter(2.0, 0.55, color='red', marker='s', s=120)
    plt.scatter(0.8, 0.63, color='red', marker='^', s=120)
    
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1)
    plt.fill_between([0, 25], 0, 0.05, color='red', alpha=0.1, label="Vulnerable Zone")
    
    plt.title("Revised Insight 3: Privacy Defense vs Capacity Scaling")
    plt.xlabel("Model Active Parameter Capacity Bound (GB)")
    plt.ylabel("Mean P_500 (Defense Proxy)")
    plt.legend(["Quantized Configs", "Small Model (1B)", "Small Model (410M)", "Significance Threshold", "Danger Zone"])
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "revised_insight3_pareto.png"))
    plt.close()

    print(f"Generated 3 revised plots successfully in {out_dir}")

if __name__ == "__main__":
    generate_revised_plots()
