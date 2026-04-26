import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset

def format_sft_entry(x, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt):
    expected_answer = x["expected_answer"]
    problem = x["problem"]
    thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
    
    final_prompt = (
        f"{reasoning_start}{thoughts}{reasoning_end}"
        f"{solution_start}{expected_answer}{solution_end}"
    )
    
    return [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]

def prepare_sft_dataset(tokenizer, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt, max_seq_length):
    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    df = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]
    
    # Filter for numeric answers
    is_number = pd.to_numeric(df["expected_answer"], errors="coerce").notnull()
    df = df[is_number].copy()
    
    df["Messages"] = df.apply(lambda x: format_sft_entry(x, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt), axis=1)
    
    # Filter by sequence length
    df["N"] = df["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    df = df[df["N"] <= max_seq_length / 2].copy()
    
    df["text"] = tokenizer.apply_chat_template(df["Messages"].values.tolist(), tokenize=False)
    return Dataset.from_pandas(df)

def prepare_grpo_dataset(tokenizer, system_prompt):
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    
    def map_grpo(x):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": x["prompt"]},
            ],
            "answer": x["solution"],
        }
    
    dataset = dataset.map(map_grpo)
    
    # Filter by length
    tokenized = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    
    return dataset, maximum_length
