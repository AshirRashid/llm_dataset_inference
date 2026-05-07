import os
import json

def get_auc(results_dir, ds, metric_key):
    try:
        with open(f"{results_dir}/{ds}_train_metrics.json", "r") as f:
            train_m = json.load(f)[metric_key]
        with open(f"{results_dir}/{ds}_val_metrics.json", "r") as f:
            val_m = json.load(f)[metric_key]
        
        y_scores = train_m + val_m
        y_true = [1]*len(train_m) + [0]*len(val_m)
        if metric_key in ["ppl", "zlib_ratio", "loss"]:
            y_scores = [-x for x in y_scores]
            
        data = list(zip(y_true, y_scores))
        # Handle ties by sorting by score, and then we will use average ranks, or just simple sort
        # For a rough AUC, simple sort is usually fine if there are few ties.
        # But wait, python's sort is stable. We can just sort.
        data.sort(key=lambda x: x[1])
        
        n1 = sum(y_true)
        n0 = len(y_true) - n1
        
        if n1 == 0 or n0 == 0:
            return None
            
        rank_sum = 0
        # 1-indexed ranks
        for i, (label, score) in enumerate(data):
            if label == 1:
                rank_sum += i + 1
                
        auc = (rank_sum - n1*(n1+1)/2) / (n1*n0)
        return max(auc, 1-auc)
    except Exception as e:
        return None

results_dir = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-12b-deduped"
metrics = {
    "loss": "ppl", 
    "mink++": "k_min_probs_0.2", 
    "zlib": "zlib_ratio"
}
datasets = ["github", "wikipedia", "arxiv", "cc"]

print("\n--- BASELINE AUC SCORES ---")
for mk_name, mk_json in metrics.items():
    aucs = []
    for ds in datasets:
        auc = get_auc(results_dir, ds, mk_json)
        if auc:
            aucs.append(auc)
    if aucs:
        print(f"pythia-12b {mk_name} AUC: {sum(aucs)/len(aucs):.4f}")

