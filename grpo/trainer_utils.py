from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from vllm import SamplingParams

def run_sft(model, tokenizer, dataset, sft_config):
    args = SFTConfig(**sft_config)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = args,
    )
    trainer.train()
    return trainer

def run_grpo(model, tokenizer, dataset, reward_funcs, max_seq_length, vllm_sampling_params, grpo_config, max_prompt_length):
    max_completion_length = max_seq_length - max_prompt_length
    
    sampling_params = SamplingParams(
        **vllm_sampling_params,
        stop = [tokenizer.eos_token],
    )
    
    training_args = GRPOConfig(
        **grpo_config,
        vllm_sampling_params = sampling_params,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
    )
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    return trainer
