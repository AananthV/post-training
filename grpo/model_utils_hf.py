import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model as get_peft_model_hf


def load_model_and_tokenizer(model_name, max_seq_length, lora_rank, gpu_memory_utilization, chat_template):
    """
    Loads a base Hugging Face model and tokenizer. Works on CPU and GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine device and dtype
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # Add compatibility methods if they don't exist (for main.py)
    if not hasattr(model, "save_lora"):
        model.save_lora = model.save_pretrained
    
    if not hasattr(model, "load_lora"):
        from peft import PeftModel
        model.load_lora = lambda path: PeftModel.from_pretrained(model, path)

    tokenizer.chat_template = chat_template
    return model, tokenizer


def get_peft_model(model, lora_rank):
    """
    Wraps the model with LoRA using the PEFT library.
    """
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model_hf(model, lora_config)
    
    # Ensure save_lora is available on the PEFT model as well
    if not hasattr(model, "save_lora"):
        model.save_lora = model.save_pretrained
        
    return model
