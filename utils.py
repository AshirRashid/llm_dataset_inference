import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantized_model_loading import load_quantized_model

# # Load tokenizer directly from the local model directory
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
# except Exception as e:
#     print(f"Failed to load tokenizer from local path {model_name}, trying base model fallback. Error: {e}")
#     tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{base_model_name}")
def prepare_model(model_name, cache_dir, quant=None):
    # Check if this is a local quantized model path
    if model_name.startswith('/scratch/ar7789/llm_quant/saved_qmodels/'):
        # Use shared utility to load model and tokenizer
        # Defaulting to cuda:0, though the utility supports passing device
        model, tokenizer = load_quantized_model(model_name, device="cuda:0")

    else:
        # For HuggingFace models
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if quant is None:
            # Full precision (float32 by default)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir, 
                trust_remote_code=True,
                device_map="auto"
            )
        elif quant == "fp16":
            # Loading in bfloat16/float16
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        elif quant == "8bit":
            # 8-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir, 
                trust_remote_code=True, 
                load_in_8bit=True,
                device_map="auto"
            )
    
    # pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 512

    print("Model loaded")
    return model, tokenizer