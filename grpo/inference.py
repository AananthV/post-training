from vllm import SamplingParams
from transformers import TextStreamer

def run_basic_inference(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    
    _ = model.generate(
        **inputs,
        temperature=0,
        max_new_tokens=1024,
        streamer=streamer,
    )

def run_fast_inference(model, tokenizer, prompt, lora_request=None):
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=50,
        max_tokens=2048,
    )
    
    # If prompt is messages list, apply template
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
    
    outputs = model.fast_generate(
        [prompt] if isinstance(prompt, str) else prompt,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    return outputs[0].outputs[0].text
