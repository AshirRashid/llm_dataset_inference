import json
import numpy as np
import os
import sys
import math
from collections import defaultdict, Counter
from scipy.stats import chi2
from datasets import load_dataset
import spacy

# Set HuggingFace cache directories to avoid permission issues with default root paths
os.environ['HF_HOME'] = '/scratch/ar7789/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/scratch/ar7789/.cache/huggingface/datasets'

# Import the project's Cohen's d function or define it if import fails
sys.path.append('/scratch/ar7789/llm_dataset_inference')
try:
    from effect_size import cohens_d
except ImportError:
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1: return float('nan')
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std

# Fisher's method for combining independent p-values
def fishers_method(p_values):
    p_values = np.array(p_values)
    p_values = np.clip(p_values, 1e-300, 1.0)
    statistic = -2 * np.sum(np.log(p_values))
    return chi2.sf(statistic, 2 * len(p_values))

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def compute_attributes_per_sample(doc, vocab, unigram_counts, total_unigrams, trigram_counts, gapped_bigram_counts, total_trigrams):
    """Computes the 6 NLP attributes for a single spaCy Doc object."""
    
    # Setup for various metrics
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
        
        # 3. Redundancy setup (get vocab IDs)
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

    # --- Final calculations ---
    
    # M1: Morphological Complexity
    M = sum(len(forms) for forms in lemma_to_surface.values()) / len(lemma_to_surface) if lemma_to_surface else 0.0
    
    # M2: Syntactic Entropy
    S = 0.0
    if total_deps > 0:
        for count in dep_counts.values():
            p = count / total_deps
            S -= p * math.log(p, 2)
            
    # M3: Redundancy (Average PMI across tokens in doc)
    R = 0.0
    trigram_samples = 0
    if len(doc_ids) >= 3:
        sum_pmi = 0.0
        for i in range(1, len(doc_ids) - 1):
            w_prev, w_curr, w_next = doc_ids[i-1], doc_ids[i], doc_ids[i+1]
            
            # Using GLOBAL probabilities for PMI calculation
            # I(w_i; w_{i-1}, w_{i+1}) = log2( P(trigram) / (P(unigram) * P(gapped_bigram)) )
            p_joint = trigram_counts[(w_prev, w_curr, w_next)] / total_trigrams
            p_wi = unigram_counts[w_curr] / total_unigrams
            p_context = gapped_bigram_counts[(w_prev, w_next)] / total_trigrams
            
            # log(P(x,y)/(P(x)P(y)))
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

def process_dataset_multivariate_new(dataset_name, model_results_path):
    print(f"\n{'='*130}")
    print(f"Dataset: {dataset_name.upper()} | Model: {os.path.basename(model_results_path)}")
    print(f"{'='*130}")

    # Load MIA metrics
    train_file = os.path.join(model_results_path, f"{dataset_name}_train_metrics.json")
    val_file = os.path.join(model_results_path, f"{dataset_name}_val_metrics.json")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Missing metrics data for {dataset_name}")
        return
        
    with open(train_file, 'r') as f: train_metrics = json.load(f)
    with open(val_file, 'r') as f: val_metrics = json.load(f)
    
    num_train = len(train_metrics[list(train_metrics.keys())[0]])
    num_val = len(val_metrics[list(val_metrics.keys())[0]])
    
    # Load HF data
    print("Loading HuggingFace dataset...")
    hf_train = load_dataset("pratyushmaini/llm_dataset_inference", name=dataset_name, split="train").select(range(num_train))
    hf_val = load_dataset("pratyushmaini/llm_dataset_inference", name=dataset_name, split="val").select(range(num_val))
    
    full_texts = [r['text'] for r in hf_train] + [r['text'] for r in hf_val]
    
    # Global Pass for counts (needed for the Redundancy metric)
    vocab = {}
    next_id = 0
    unigram_counts = Counter()
    trigram_counts = Counter()
    gapped_bigram_counts = Counter()
    total_unigrams = 0
    total_trigrams = 0
    
    print("Global pass for frequency counts...")
    processed_docs = []
    # Using nlp.pipe for efficiency
    for doc in nlp.pipe(full_texts):
        processed_docs.append(doc)
        doc_ids = []
        for token in doc:
            w_lower = token.text.lower()
            if w_lower not in vocab:
                vocab[w_lower] = next_id
                next_id += 1
            w_id = vocab[w_lower]
            doc_ids.append(w_id)
            unigram_counts[w_id] += 1
            total_unigrams += 1
            
        for i in range(1, len(doc_ids) - 1):
            w_prev, w_curr, w_next = doc_ids[i-1], doc_ids[i], doc_ids[i+1]
            trigram_counts[(w_prev, w_curr, w_next)] += 1
            gapped_bigram_counts[(w_prev, w_next)] += 1
            total_trigrams += 1
            
    # Per-sample attribute pass
    print(f"Computing attributes for {len(full_texts)} samples...")
    sample_attrs = []
    for doc in processed_docs:
        attrs = compute_attributes_per_sample(doc, vocab, unigram_counts, total_unigrams, trigram_counts, gapped_bigram_counts, total_trigrams)
        sample_attrs.append(attrs)
        
    train_attrs = sample_attrs[:num_train]
    val_attrs = sample_attrs[num_train:]
    
    attribute_names = list(train_attrs[0].keys())
    
    # Quantile analysis
    for attr in attribute_names:
        print(f"\n--- Stratifying by: {attr} ---")
        train_vals = np.array([s[attr] for s in train_attrs])
        val_vals = np.array([s[attr] for s in val_attrs])
        
        # Calculate quartiles based on train set
        quartiles = np.percentile(train_vals, [25, 50, 75])
        
        print(f"{'Quartile':<15} | {'Avg Attr':<15} | {'Best Metric':<35} | {'Best Cohen D':<12}")
        print("-" * 100)
        
        for q in range(4):
            if q == 0:
                train_idx = np.where(train_vals <= quartiles[0])[0]
                val_idx = np.where(val_vals <= quartiles[0])[0]
                label = "Q1 (Lowest)"
            elif q == 1:
                train_idx = np.where((train_vals > quartiles[0]) & (train_vals <= quartiles[1]))[0]
                val_idx = np.where((val_vals > quartiles[0]) & (val_vals <= quartiles[1]))[0]
                label = "Q2"
            elif q == 2:
                train_idx = np.where((train_vals > quartiles[1]) & (train_vals <= quartiles[2]))[0]
                val_idx = np.where((val_vals > quartiles[1]) & (val_vals <= quartiles[2]))[0]
                label = "Q3"
            else:
                train_idx = np.where(train_vals > quartiles[2])[0]
                val_idx = np.where(val_vals > quartiles[2])[0]
                label = "Q4 (Highest)"
                
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
                
            best_cd = -1
            best_metric = ""
            actual_cd = 0
            
            for metric in train_metrics.keys():
                t_arr = np.array(train_metrics[metric])[train_idx]
                v_arr = np.array(val_metrics[metric])[val_idx]
                
                # More negative = more leakage usually in this pipeline context, 
                # but cohens_d(v, t) will be positive if train loss is lower.
                cd = cohens_d(v_arr, t_arr)
                if abs(cd) > best_cd:
                    best_cd = abs(cd)
                    actual_cd = cd
                    best_metric = metric
                    
            avg_attr = np.mean(train_vals[train_idx])
            print(f"{label:<15} | {avg_attr:<15.4f} | {best_metric:<35} | {actual_cd:<12.4f}")

if __name__ == "__main__":
    datasets = ["arxiv", "wikipedia", "github", "cc"]
    # Target the baseline non-quantized model for this new discovery phase
    model_path = "/scratch/ar7789/llm_dataset_inference/results/EleutherAI/pythia-12b-deduped"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        sys.exit(1)
        
    for ds in datasets:
        process_dataset_multivariate_new(ds, model_path)
