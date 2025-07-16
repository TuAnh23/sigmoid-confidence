from datasets import Dataset
from transformers import AutoTokenizer
import torch

def line_pairs(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            yield {"input": src.strip(), "target": tgt.strip()}

# Format: instruction + response
def format_example(example, tokenizer, model_id):
    if model_id == "Unbabel/TowerInstruct-7B-v0.2":
        messages = [
            {"role": "user", "content": f"Translate the following text from English into German.\nEnglish: {example['input']}.\nGerman:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt + example['target']
    else:
        raise NotImplementedError

    return {"text": prompt}

# Tokenize with labels (shifted targets)
def tokenize_fn(example, tokenizer, max_length=1024):
    # We treat the full prompt+response as a causal LM sequence
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Main function to build datasets
def build_datasets(tokenizer, model_id, max_length=2048):
    data_dir = "/hkfs/home/project/hk-project-p0021679/ho6084/fairseq/examples/confidence_aware_ssl/data/ParaCrawl"
    # Wrap with Hugging Face datasets
    train_dataset = Dataset.from_generator(lambda: line_pairs(f"{data_dir}/train.5M.dedup.en", f"{data_dir}/train.5M.dedup.de"))
    eval_dataset  = Dataset.from_generator(lambda: line_pairs(f"{data_dir}/dev.en", f"{data_dir}/dev.de"))

    # Format with prompt
    train_dataset = train_dataset.map(
        lambda x: format_example(x, tokenizer, model_id),
        load_from_cache_file=True,
        num_proc=100
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_example(x, tokenizer, model_id),
        load_from_cache_file=True,
        num_proc=100
    )

    # Tokenize
    train_dataset = train_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        load_from_cache_file=True,
        remove_columns=train_dataset.column_names,
        num_proc=100
    )
    eval_dataset = eval_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        load_from_cache_file=True,
        remove_columns=eval_dataset.column_names,
        num_proc=100
    )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, eval_dataset
