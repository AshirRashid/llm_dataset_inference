import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

def compute_ppl(model, tokenizer, dataset_name, split="train", num_samples=1000):
    print(f"\nProcessing {dataset_name}...")
    ds = load_dataset("pratyushmaini/llm_dataset_inference", name=dataset_name, split=split)
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))
    
    losses = []
    model.eval()
    
    for item in tqdm(ds, desc=f"Computing loss for {dataset_name}"):
        text = item['text']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            losses.append(loss)
            
    return np.mean(losses)

def main():
    model_name = "EleutherAI/pythia-12b-deduped"
    cache_dir = "/scratch/ar7789/.cache/huggingface"
    
    print(f"Loading {model_name} in Dynamic 8-bit mode...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Using load_in_8bit=True as per Dynamic 8-bit definition
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        load_in_8bit=True,
        device_map="auto"
    )
    
    datasets = ["arxiv", "wikipedia", "github", "cc"]
    all_losses = []
    
    results = {}
    for ds_name in datasets:
        mean_loss = compute_ppl(model, tokenizer, ds_name)
        # PPL for a subset is exp(mean_loss)
        results[ds_name] = np.exp(mean_loss)
        all_losses.append(mean_loss)
        print(f"Perplexity for {ds_name}: {results[ds_name]:.4f}")
        
    # The final report PPL is the mean PPL across the four subsets
    # (Or the exp of the mean loss across all tokens, which is equivalent if weights are equal)
    avg_ppl = np.mean([results[ds] for ds in datasets])
    print(f"\n" + "="*40)
    print(f"RESULTS FOR {model_name} (DYNAMIC 8-BIT)")
    print("="*40)
    for ds, val in results.items():
        print(f"{ds:12}: {val:.4f}")
    print("-" * 40)
    print(f"Final Mean Perplexity: {avg_ppl:.4f}")
    print("="*40)
    
    # Save results to absolute path
    output_path = "/scratch/ar7789/llm_dataset_inference/results/dynamic_8bit_ppl_results.json"
    results["mean_perplexity"] = avg_ppl
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
