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
                "Log10 P_500": math.log10(mean_p) if mean_p > 0 else -100,
            })
    return records

def generate_plots():
    records = load_data()
    out_dir = "/scratch/ar7789/llm_dataset_inference/results_analysis/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    colors = {"Baseline": "red", "AWQ": "blue", "Static": "green", "GPTQ": "orange"}

    # --- Insight 1 & 2 ---
    plt.figure(figsize=(14, 8))
    ds_avg = defaultdict(list)
    for r in records: ds_avg[r["Dataset"]].append(r["Log10 P_500"])
    sorted_datasets = sorted(ds_avg.keys(), key=lambda d: sum(ds_avg[d])/len(ds_avg[d]))
    
    for i, ds in enumerate(sorted_datasets):
        ds_recs = [r for r in records if r["Dataset"] == ds]
        for r in ds_recs:
            plt.scatter(i, r["Log10 P_500"], color=colors[r["Family"]], alpha=0.6, s=50)

    plt.xticks(range(len(sorted_datasets)), sorted_datasets, rotation=45, ha="right")
    plt.title("Insight 1 & 2: Massive Dataset Variance vs Tight Quantization Variance")
    plt.ylabel("Log10(Mean P_500) - Lower is More Vulnerable")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "insight1_dataset_variance.png"))
    plt.close()

    # --- Insight 3 ---
    arxiv_recs = [r for r in records if r["Dataset"] == "arxiv" and r["Base Model"] == "Pythia-12B"]
    arxiv_recs.sort(key=lambda x: x["Mean P_500"])
    
    plt.figure(figsize=(12, 10))
    plt.barh([r["Config"] for r in arxiv_recs], [r["Mean P_500"] for r in arxiv_recs], color=[colors[r["Family"]] for r in arxiv_recs])
    plt.title("Insight 3: Extreme Quantization (GPTQ-2bit) Suppresses Privacy Leakage (Arxiv)")
    plt.xlabel("Mean P_500 (Higher = Less Vulnerable to Inference)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "insight3_quantization_extremes.png"))
    plt.close()

    # --- Insight 4 ---
    plt.figure(figsize=(10, 6))
    
    # Existing 12B arxiv stats mapped to memory sizes
    mem_12b_points = []
    for r in arxiv_recs:
        mem = 10.0
        if r["Family"] == "Baseline": mem = 24.0
        elif "b4" in r["Config"] or "4bit" in r["Config"]: mem = 6.0
        elif "b3" in r["Config"] or "3bit" in r["Config"]: mem = 4.5
        elif "b2" in r["Config"] or "2bit" in r["Config"]: mem = 3.0
        else: mem = 12.0
        plt.scatter(mem, r["Mean P_500"], color=colors[r["Family"]], marker='o', alpha=0.7, s=80)
        
    # Baseline 1B (2GB) and 410M (0.8GB) data on Arxiv manually inputted
    plt.scatter(2.0, 0.55, color='red', marker='s', s=120, label='Pythia-1B Baseline')
    plt.scatter(0.8, 0.63, color='red', marker='^', s=120, label='Pythia-410M Baseline')
    
    plt.title("Insight 4: Model Memory Footprint vs Privacy Leakage (Arxiv)")
    plt.xlabel("Estimated Model Memory Footprint (GB)")
    plt.ylabel("Mean P_500")
    plt.legend(["12B Configurations (varies)", "Pythia-1B (Unquantized)", "Pythia-410M (Unquantized)"])
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "insight4_memory_vs_leakage.png"))
    plt.close()

    print(f"Generated 3 plots successfully in {out_dir}")

if __name__ == "__main__":
    generate_plots()
