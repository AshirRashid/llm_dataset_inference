import json
import numpy as np
import os
import sys
import math
import glob
from collections import defaultdict, Counter
from datasets import load_dataset
import spacy
# Ensure local imports work when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import cohens_d, fishers_method

# Environment setup
os.environ['HF_HOME'] = '/scratch/ar7789/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/scratch/ar7789/.cache/huggingface/datasets'

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def compute_attributes_per_sample(doc, vocab, unigram_counts, total_unigrams, trigram_counts, gapped_bigram_counts, total_trigrams):
    """Computes the 6 NLP attributes for a single spaCy Doc object."""
    lemma_to_surface = defaultdict(set)
    total_word_len = 0
    total_capitalized = 0
    total_words = 0
    unique_words_lower = set()
    dep_counts = Counter()
    total_deps = 0
    doc_ids = []
    
    for token in doc:
        # 1. Morphological Complexity
        lemma_to_surface[token.lemma_].add(token.text)
        # 2. Syntactic Entropy
        dep_counts[token.dep_] += 1
        total_deps += 1
        # 3. Redundancy setup
        w_lower = token.text.lower()
        if w_lower in vocab:
            doc_ids.append(vocab[w_lower])
        # 4, 5, 6 setup
        if not token.is_punct and not token.is_space:
            total_words += 1
            total_word_len += len(token.text)
            if token.text and token.text[0].isupper():
                total_capitalized += 1
            unique_words_lower.add(w_lower)

    # M1: Morphological Complexity
    M = sum(len(forms) for forms in lemma_to_surface.values()) / len(lemma_to_surface) if lemma_to_surface else 0.0
    # M2: Syntactic Entropy
    S = 0.0
    if total_deps > 0:
        for count in dep_counts.values():
            p = count / total_deps
            S -= p * math.log(p, 2)
    # M3: Redundancy
    R = 0.0
    trigram_samples = 0
    if len(doc_ids) >= 3:
        sum_pmi = 0.0
        for i in range(1, len(doc_ids) - 1):
            w_prev, w_curr, w_next = doc_ids[i-1], doc_ids[i], doc_ids[i+1]
            p_joint = trigram_counts[(w_prev, w_curr, w_next)] / total_trigrams
            p_wi = unigram_counts[w_curr] / total_unigrams
            p_context = gapped_bigram_counts[(w_prev, w_next)] / total_trigrams
            pmi = math.log(p_joint / (p_wi * p_context), 2)
            sum_pmi += pmi
            trigram_samples += 1
        R = sum_pmi / trigram_samples if trigram_samples > 0 else 0.0
    # M4: Avg Word Length
    T = total_word_len / total_words if total_words > 0 else 0.0
    # M5: Capitalization Rate
    C = total_capitalized / total_words if total_words > 0 else 0.0
    # M6: Vocabulary Richness
    D = len(unique_words_lower) / total_words if total_words > 0 else 0.0
    
    return {
        "Morphological Complexity": M,
        "Syntactic Entropy": S,
        "Redundancy": R,
        "Avg Word Length": T,
        "Capitalization Rate": C,
        "Vocabulary Richness": D
    }

def precompute_dataset_attributes(dataset_name, num_train, num_val):
    print(f"Pre-computing NLP attributes for {dataset_name.upper()} ({num_train} train, {num_val} val)...")
    hf_train = load_dataset("pratyushmaini/llm_dataset_inference", name=dataset_name, split="train").select(range(num_train))
    hf_val = load_dataset("pratyushmaini/llm_dataset_inference", name=dataset_name, split="val").select(range(num_val))
    full_texts = [r['text'] for r in hf_train] + [r['text'] for r in hf_val]
    
    vocab, next_id = {}, 0
    unigram_counts, trigram_counts, gapped_bigram_counts = Counter(), Counter(), Counter()
    total_unigrams, total_trigrams = 0, 0
    
    processed_docs = []
    for doc in nlp.pipe(full_texts):
        processed_docs.append(doc)
        doc_ids = []
        for token in doc:
            w_l = token.text.lower()
            if w_l not in vocab: vocab[w_l] = next_id; next_id += 1
            w_id = vocab[w_l]
            doc_ids.append(w_id)
            unigram_counts[w_id] += 1
            total_unigrams += 1
        for i in range(1, len(doc_ids) - 1):
            trigram_counts[(doc_ids[i-1], doc_ids[i], doc_ids[i+1])] += 1
            gapped_bigram_counts[(doc_ids[i-1], doc_ids[i+1])] += 1
            total_trigrams += 1
            
    sample_attrs = [compute_attributes_per_sample(doc, vocab, unigram_counts, total_unigrams, trigram_counts, gapped_bigram_counts, total_trigrams) for doc in processed_docs]
    return sample_attrs[:num_train], sample_attrs[num_train:]

def main():
    datasets = ["arxiv", "wikipedia", "github", "cc"]
    # Results directory mapping (Absolute paths)
    base_results_dir = "/scratch/ar7789/llm_dataset_inference/results"
    quant_results_dir = os.path.join(base_results_dir, "scratch/ar7789/llm_quant/saved_qmodels/tmp/EleutherAI")
    
    # 1. Identify all models
    models = [os.path.join(base_results_dir, "EleutherAI/pythia-12b-deduped")]
    models += glob.glob(os.path.join(quant_results_dir, "pythia-12b-deduped-*"))
    models = sorted([m for m in models if os.path.isdir(m)])
    
    print(f"Found {len(models)} models for analysis.")
    
    # 2. Pre-compute attributes for each dataset (Scope: 2000 samples per split)
    dataset_attrs = {}
    for ds in datasets:
        dataset_attrs[ds] = precompute_dataset_attributes(ds, 2000, 2000)
        
    # 3. Iterate over models and datasets
    final_results = {}
    
    for model_path in models:
        model_name = os.path.basename(model_path)
        final_results[model_name] = {}
        print(f"\nProcessing Model: {model_name}")
        
        for ds in datasets:
            train_file = os.path.join(model_path, f"{ds}_train_metrics.json")
            val_file = os.path.join(model_path, f"{ds}_val_metrics.json")
            if not os.path.exists(train_file): continue
            
            with open(train_file, 'r') as f: t_metrics = json.load(f)
            with open(val_file, 'r') as f: v_metrics = json.load(f)
            
            t_attrs, v_attrs = dataset_attrs[ds]
            # Match lengths
            min_t = min(len(t_attrs), len(t_metrics[list(t_metrics.keys())[0]]))
            min_v = min(len(v_attrs), len(v_metrics[list(v_metrics.keys())[0]]))
            
            ds_res = {}
            for attr in t_attrs[0].keys():
                t_vals = np.array([s[attr] for s in t_attrs[:min_t]])
                v_vals = np.array([s[attr] for s in v_attrs[:min_v]])
                quartiles = np.percentile(t_vals, [25, 50, 75])
                
                attr_res = []
                for q in range(4):
                    if q == 0: t_idx, v_idx = np.where(t_vals <= quartiles[0])[0], np.where(v_vals <= quartiles[0])[0]
                    elif q == 1: t_idx, v_idx = np.where((t_vals > quartiles[0]) & (t_vals <= quartiles[1]))[0], np.where((v_vals > quartiles[0]) & (v_vals <= quartiles[1]))[0]
                    elif q == 2: t_idx, v_idx = np.where((t_vals > quartiles[1]) & (t_vals <= quartiles[2]))[0], np.where((v_vals > quartiles[1]) & (v_vals <= quartiles[2]))[0]
                    else: t_idx, v_idx = np.where(t_vals > quartiles[2])[0], np.where(v_vals > quartiles[2])[0]
                    
                    if len(t_idx) == 0 or len(v_idx) == 0: continue
                    
                    best_cd, best_metric = -1, ""
                    for m in t_metrics.keys():
                        cd = cohens_d(np.array(v_metrics[m])[v_idx], np.array(t_metrics[m])[t_idx])
                        if abs(cd) > best_cd: best_cd, best_metric = abs(cd), m
                    
                    attr_res.append({"quartile": q+1, "avg_attr": float(np.mean(t_vals[t_idx])), "best_metric": best_metric, "cohens_d": best_cd})
                ds_res[attr] = attr_res
            final_results[model_name][ds] = ds_res

    # 4. Save results in the script's directory using absolute paths
    results_path = "/scratch/ar7789/llm_dataset_inference/scripts/dataset_stratified_nlp_x_quant_x_mia/all_models/analysis_results.json"
    report_path = "/scratch/ar7789/llm_dataset_inference/scripts/dataset_stratified_nlp_x_quant_x_mia/all_models/report.txt"
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    # 5. Generate a simple summary report
    with open(report_path, "w") as f:
        f.write("Multivariate Analysis Report: Quantization Resilience\n")
        f.write("="*60 + "\n\n")
        for model in sorted(final_results.keys()):
            f.write(f"Model: {model}\n")
            for ds in datasets:
                if ds not in final_results[model]: continue
                f.write(f"  Dataset: {ds.upper()}\n")
                for attr, quartiles in final_results[model][ds].items():
                    f.write(f"    - {attr}: ")
                    cds = [f"Q{q['quartile']}:{q['cohens_d']:.3f}" for q in quartiles]
                    f.write(", ".join(cds) + "\n")
            f.write("\n")

if __name__ == "__main__":
    main()
