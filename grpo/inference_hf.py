from transformers import TextStreamer
import torch

def run_basic_inference(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    device = model.device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    
    _ = model.generate(
        **inputs,
        temperature=0.7,
        max_new_tokens=1024,
        streamer=streamer,
    )

def run_fast_inference(model, tokenizer, prompt, lora_request=None):
    """
    Standard HuggingFace generation (CPU/GPU compatible).
    """
    # If prompt is messages list, apply template
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
    
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Standard HF generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=1.0,
        top_k=50,
    )
    
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
