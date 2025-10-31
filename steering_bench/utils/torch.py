import gc
import torch

from steering_bench.core.types import Model, Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_with_quantization(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple[Model, Tokenizer]:
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Load the model with quantization if specified
    if load_in_4bit or load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = load_in_4bit,
            load_in_8bit = load_in_8bit,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_quant_type = "nf4")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        
    return model, tokenizer


class EmptyTorchCUDACache:
    """Context manager to free GPU memory"""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
