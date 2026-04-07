from utils import prepare_model
from metrics import aggregate_metrics, reference_model_registry
import json, os
import argparse
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m-deduped", help='The name of the model to use')
    parser.add_argument('--dataset_name', type=str, default="wikipedia", help='The name of the dataset to use')
    parser.add_argument('--split', type=str, default="train", help='The split of the dataset to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to use')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size to use')
    parser.add_argument('--from_hf', type=int, default=1, help='If set, will load the dataset from huggingface')
    parser.add_argument('--cache_dir', type=str, default="/scratch/ar7789/.cache/huggingface", help='The directory to cache the model')
    parser.add_argument('--quant', type=str, choices=["fp16", "8bit"], help='Quantization type to use for loading')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    results_file = f"results/{args.model_name}/{args.dataset_name}_{args.split}_metrics.json"
    # if os.path.exists(results_file):
        # print(f"Results file {results_file} already exists. Aborting...")
        # return
    model_name =  args.model_name
    
    if model_name in ["microsoft/phi-1_5", "EleutherAI/pythia-12b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-410m"]:
        args.cache_dir = "/scratch/ar7789/.cache/huggingface/pratyush"

    model, tokenizer = prepare_model(model_name, cache_dir= args.cache_dir, quant=args.quant)

    print("Model Loaded:\n", model)
    print("Tokenizer Loaded:\n", tokenizer)

    # load the data
    dataset_name = args.dataset_name
    split = args.split

    from datasets import load_dataset
    dataset = load_dataset("pratyushmaini/llm_dataset_inference", name = dataset_name, split = split)
    
    # Limit to num_samples if specified
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
    
    # if args.from_hf:
    #     # Load dataset from Hugging Face
    #     dataset = load_dataset(dataset_name, split=split)
    #     # Limit to num_samples if specified
    #     if args.num_samples and args.num_samples < len(dataset):
    #         dataset = dataset.select(range(args.num_samples))
    # else:
    #     from dataloader import load_data
    #     # if you want to load data directly from the PILE, use the following line
    #     num_samples = args.num_samples
    #     dataset = load_data(dataset_name, split, num_samples)

    print("Data loaded")

    # get the metrics
    if model_name in reference_model_registry.values():
        metric_list = ["ppl"]
        print(f"Model {model_name} is a reference model, only calculating perplexity")
    else:
        metric_list = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs", "perturbation", "reference_model"]
        print(f"Model {model_name} is not a reference model, calculating full metric suite: {metric_list}")
    
    print(f"Calculating metrics for model: {model_name}")
    print(f"Dataset: {dataset_name}, Split: {split}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    print(f"First sample keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'No samples'}")
    
    metrics = aggregate_metrics(model, tokenizer, dataset, metric_list, args, batch_size = args.batch_size)
    print(f"Successfully calculated metrics: {list(metrics.keys())}")
    
    # save the metrics
    os.makedirs(f"results/{model_name}", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()

