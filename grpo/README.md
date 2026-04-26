# Qwen3-4B GRPO Training Project

This repository is a modularized version of the Qwen3-4B GRPO training notebook. It uses Unsloth for efficient training and vLLM for fast inference.

## Project Structure

- `config.py`: Contains all configuration parameters for models, prompts, and training.
- `model_utils.py`: Logic for loading the base model and setting up PEFT (LoRA).
- `data_utils.py`: Functions for loading and formatting the SFT and GRPO datasets.
- `rewards.py`: Reward functions used by the GRPOTrainer to guide the model's reasoning.
- `trainer_utils.py`: Wrappers for SFT and GRPO training loops.
- `inference.py`: Utilities for running inference with the trained model.
- `main.py`: The main entry point that orchestrates the entire training and evaluation process.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the full training pipeline:

```bash
python main.py
```

## Features

- **Pre-fine-tuning (SFT)**: Primes the model to follow the `<start_working_out>` and `<SOLUTION>` formatting.
- **GRPO Training**: Uses Group Relative Policy Optimization to improve mathematical reasoning.
- **Reward Functions**: Includes exact format matching, approximate format matching, and numeric answer checking.
- **vLLM Integration**: Enables fast generation during GRPO training and inference.
