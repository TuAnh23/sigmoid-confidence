from datasets import Dataset
from transformers import AutoTokenizer
import torch
import os

def line_pairs(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            yield {"input": src.strip(), "target": tgt.strip()}

# Format: instruction + response
def format_and_tokenize_example(example, tokenizer, max_length=1024):
    full_messages = [
        {"role": "user", "content": f"Translate the following text from English into German.\nEnglish: {example['input']}.\nGerman:"},
        {"role": "assistant", "content": example['target']}
    ]
    input_message = [full_messages[0]]
    tokenized_full_prompt = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=True, max_length=max_length, padding="max_length", truncation=True, return_dict=True)
    input_prompt_length = len(tokenizer.apply_chat_template(input_message, tokenize=True, add_generation_prompt=True))

    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

    # Mask out input prompt in the labels so we dont train on it later
    tokenized_full_prompt["labels"][:input_prompt_length] = [tokenizer.pad_token_id] * input_prompt_length

    return tokenized_full_prompt


# Main function to build datasets
def build_datasets(tokenizer, max_length=1024):
    data_dir = f"{os.environ.get('ROOT_DIR')}/fairseq/examples/confidence_aware_ssl/data/ParaCrawl"
    # Wrap with Hugging Face datasets
    train_dataset = Dataset.from_generator(lambda: line_pairs(f"{data_dir}/train.5M.dedup.en", f"{data_dir}/train.5M.dedup.de"))
    eval_dataset  = Dataset.from_generator(lambda: line_pairs(f"{data_dir}/dev.en", f"{data_dir}/dev.de"))

    train_dataset = train_dataset.map(
        lambda x: format_and_tokenize_example(x, tokenizer, max_length),
        load_from_cache_file=True,
        num_proc=100
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_and_tokenize_example(x, tokenizer, max_length),
        load_from_cache_file=True,
        num_proc=100
    )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, eval_dataset
