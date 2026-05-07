import json
import os
import csv

models = {
    "awq_b4-gs32-zp1": "/scratch/ar7789/llm_dataset_inference/results/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-awq-b4-gs32-zp1",
    "gptq_b4-gs128-da0": "/scratch/ar7789/llm_dataset_inference/results/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b4-gs128-da0",
    "gptq_b8-gs-1-da1": "/scratch/ar7789/llm_dataset_inference/results/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-gptq-b8-gs-1-da1",
    "static_4bit-fp4-bfloat16": "/scratch/ar7789/llm_dataset_inference/results/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-static-4bit-fp4-bfloat16",
    "dynamic_8bit": "/scratch/ar7789/llm_dataset_inference/results/scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI/pythia-12b-deduped-dynamic-b8",
    "pythia-12b-deduped (unquantized)": "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-12b-deduped"
}

datasets = ["github", "wikipedia", "arxiv", "cc"]
splits = ["train", "val"]

print("--- PERPLEXITY ---")
for name, path in models.items():
    ppls = []
    for ds in datasets:
        for sp in splits:
            file_path = f"{path}/{ds}_{sp}_metrics.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        if "ppl" in data:
                            ppls.extend(data["ppl"])
                    except:
                        pass
    if ppls:
        print(f"{name}: {sum(ppls)/len(ppls):.4f} (N={len(ppls)})")
    else:
        print(f"{name}: No data found")

print("\n--- BASELINE MIA SCORES (Dataset MIA p-values) ---")
p_value_dir = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/mean+p-value-outliers/train-normalize/EleutherAI_pythia-12b-deduped"
if os.path.exists(p_value_dir):
    p500s_all = []
    for ds in datasets:
        fpath = os.path.join(p_value_dir, f"{ds}.csv")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                reader = csv.DictReader(f)
                p500s = [float(row["p_500"]) for row in reader if "p_500" in row]
                if p500s:
                    mean_p500 = sum(p500s)/len(p500s)
                    print(f"pythia-12b {ds} p_500: {mean_p500:.4e}")
                    p500s_all.append(mean_p500)
    if p500s_all:
        print(f"Average baseline p_500 across datasets: {sum(p500s_all)/len(p500s_all):.4e}")

