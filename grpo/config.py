# -*- coding: utf-8 -*-

class Config:
    # Model configuration
    model_name = "unsloth/Qwen3-4B-Base"
    max_seq_length = 2048
    lora_rank = 32
    gpu_memory_utilization = 0.9
    
    # Prompt formatting
    reasoning_start = "<start_working_out>"
    reasoning_end   = "<end_working_out>"
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"
    
    system_prompt = (
        f"You are given a problem.\n"
        f"Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )
    
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '" + system_prompt + "' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
        "{% endif %}"
    )

    # SFT Training Configuration
    sft_config = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 5,
        "num_train_epochs": 2,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "report_to": "none",
    }

    # GRPO Training Configuration
    grpo_config = {
        "learning_rate": 5e-6,
        "weight_decay": 0.001,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_generations": 4,
        "max_steps": 100,
        "save_steps": 100,
        "report_to": "none",
        "output_dir": "outputs",
    }

    # vLLM Sampling Parameters
    vllm_sampling_params = {
        "min_p": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 3407,
        "include_stop_str_in_output": True,
    }
