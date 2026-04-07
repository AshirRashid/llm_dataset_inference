import csv
import glob
import os
import re

CSV_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/mean+p-value-outliers/train-normalize"

def simplify_model_name(subdir: str) -> str:
    if "dynamic" in subdir:
        if "8bit" in subdir:
            return "dynamic-8bit"
    if "static" in subdir:
        if "4bit" in subdir and "fp4" in subdir and "bfloat16" in subdir:
            return "static-4bit-fp4-bfloat16"
        if "8bit" in subdir and "ns256" in subdir:
            return "static-8bit-ns256"
        parts = subdir.split("static-")
        if len(parts) > 1:
            return "static-" + parts[-1]

    m = re.search(r'(gptq|awq)-(.*)', subdir)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    
    return subdir

def main():
    target_models = ['12b', '1b', '410m', '160m']
    # Glob for all dataset CSVs inside the model directories
    all_files = glob.glob(os.path.join(CSV_DIR, "*", "*.csv"))
    
    files_to_process = []
    for f in all_files:
        parent_dir = os.path.basename(os.path.dirname(f))
        if any(f"pythia-{m}" in parent_dir for m in target_models):
            files_to_process.append(f)
    
    if not files_to_process:
        print(f"No CSV files found for models {target_models} in {CSV_DIR}.")
        return

    print(f"Found {len(files_to_process)} dataset CSVs corresponding to targeted models.\n")

    results = []

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        dataset = filename.replace('.csv', '')
        
        subdir = os.path.basename(os.path.dirname(filepath))
        model_size = next((m for m in target_models if m in subdir), "unknown")
        
        p_list = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            if 'p_500' not in reader.fieldnames:
                print(f"Skipping {filepath}: no p_500 column")
                continue
                
            for row in reader:
                p_num_str = row.get('p_500')
                try:
                    p_num = float(p_num_str)
                    if p_num == p_num:  # nan check
                        p_list.append(p_num)
                except (ValueError, TypeError):
                    continue

        if p_list:
            mean_p = sum(p_list) / len(p_list)
            simple_name = simplify_model_name(subdir)
            results.append({
                "Base Model": f"Pythia-{model_size.upper()}",
                "Dataset": dataset,
                "Config": simple_name,
                "Original_Subdir": subdir,
                "Mean P_500": mean_p
            })
            
    if results:
        results.sort(key=lambda x: (x["Base Model"], x["Dataset"], x["Mean P_500"]))
        
        unique_configs = set(x["Config"] for x in results)
        print("=== UNIQUE CONFIGS DISCOVERED ===")
        for c in sorted(unique_configs):
            print(c)
            
        print("\n=== TOP 20 LOWEST P-VALUES (Dataset Inference Signal) ===")
        header = f"{'Base Model':<15} | {'Dataset':<10} | {'Config':<30} | {'Mean P_500':<15}"
        print(header)
        print("-" * len(header))
        for r in results[:20]:
            print(f"{r['Base Model']:<15} | {r['Dataset']:<10} | {r['Config']:<30} | {r['Mean P_500']:.6e}")
        
        out_csv = "/scratch/ar7789/llm_dataset_inference/results_analysis/p_500_summary.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Base Model", "Dataset", "Config", "Original_Subdir", "Mean P_500"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved full results to {out_csv}")
        
        # Output tabular format matching LaTeX
        out_md = "/scratch/ar7789/llm_dataset_inference/results_analysis/report_table.md"
        with open(out_md, "w") as f:
            f.write("### Markdown Format\n")
            f.write("| Base Model | Dataset | Config | Mean P_500 |\n")
            f.write("|------------|---------|--------|------------|\n")
            for r in results:
                f.write(f"| {r['Base Model']} | {r['Dataset']} | {r['Config']} | {r['Mean P_500']:.6e} |\n")
                
            f.write("\n### LaTeX Format\n")
            f.write("\\begin{table}[tb]\n\\centering\n\\begin{tabular}{llll}\n\\hline\n")
            f.write("\\textbf{Base Model} & \\textbf{Dataset} & \\textbf{Config} & \\textbf{P-Value} \\\\\n\\hline\n")
            for r in results:
                f.write(f"{r['Base Model']} & {r['Dataset']} & {r['Config']} & {r['Mean P_500']:.6e} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()
