import torch
import gc
from config import Config
from model_utils import load_model_and_tokenizer, get_peft_model
from data_utils import prepare_sft_dataset, prepare_grpo_dataset
from rewards import RewardManager
from trainer_utils import run_sft, run_grpo
from inference import run_fast_inference

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    config = Config()
    
    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        config.max_seq_length,
        config.lora_rank,
        config.gpu_memory_utilization,
        config.chat_template
    )
    model = get_peft_model(model, config.lora_rank)
    
    # 2. Pre-fine-tuning (SFT)
    print("Preparing SFT dataset...")
    sft_dataset = prepare_sft_dataset(
        tokenizer,
        config.reasoning_start,
        config.reasoning_end,
        config.solution_start,
        config.solution_end,
        config.system_prompt,
        config.max_seq_length
    )
    
    print("Running SFT...")
    run_sft(model, tokenizer, sft_dataset, config.sft_config)
    
    # Cleanup after SFT
    del sft_dataset
    cleanup()
    
    # 3. GRPO Training
    print("Preparing GRPO dataset...")
    grpo_dataset, max_prompt_length = prepare_grpo_dataset(tokenizer, config.system_prompt)
    
    reward_manager = RewardManager(
        tokenizer,
        config.reasoning_end,
        config.solution_start,
        config.solution_end
    )
    reward_funcs = reward_manager.get_reward_funcs()
    
    print("Running GRPO...")
    run_grpo(
        model,
        tokenizer,
        grpo_dataset,
        reward_funcs,
        config.max_seq_length,
        config.vllm_sampling_params,
        config.grpo_config,
        max_prompt_length
    )
    
    # 4. Save and Test
    print("Saving LoRA...")
    model.save_lora("grpo_saved_lora")
    
    print("Testing inference...")
    test_prompt = "What is the sqrt of 101?"
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user",   "content": test_prompt},
    ]
    
    response = run_fast_inference(
        model, 
        tokenizer, 
        messages, 
        lora_request=model.load_lora("grpo_saved_lora")
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
