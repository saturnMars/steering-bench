import gc
import torch

from steering_bench.core.types import Model, Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_with_quantization(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device = None,
) -> tuple[Model, Tokenizer]:
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Set device map
    device_map = "auto" if not device or device == -1 else {'':device}
    
    # Load the model with quantization if specified
    if load_in_4bit or load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = load_in_4bit,
            load_in_8bit = load_in_8bit,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_quant_type = "nf4")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, dtype = torch.bfloat16)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Print model info 
    print(f"\nLLM: {model_name} ({model.device}, {'quantized' if hasattr(model, 'is_quantized') else model.dtype})\n")

    return model, tokenizer


class EmptyTorchCUDACache:
    """Context manager to free GPU memory"""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
