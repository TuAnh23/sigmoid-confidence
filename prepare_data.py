from datasets import Dataset
from transformers import AutoTokenizer
import torch
import os

def line_pairs(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            yield {"input": src.strip(), "target": tgt.strip()}

# Format for teacher forcing: instruction + response
def format_and_tokenize_example_for_teacher_forcing(example, src_lang, tgt_lang, tokenizer, max_length=1024):
    full_messages = [
        {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['input']}.\n{tgt_lang}:"},
        {"role": "assistant", "content": example['target']}
    ]
    input_message = [full_messages[0]]
    tokenized_full_prompt = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=True, max_length=max_length, padding="max_length", truncation=True, return_dict=True)
    input_prompt_length = len(tokenizer.apply_chat_template(input_message, tokenize=True, add_generation_prompt=True))

    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

    # Mask out input prompt in the labels so we dont train on it later
    tokenized_full_prompt["labels"][:input_prompt_length] = [tokenizer.pad_token_id] * input_prompt_length

    return tokenized_full_prompt


# Format for batched inference: instruction only
def format_and_tokenize_example_for_inference(example, src_lang, tgt_lang, tokenizer, max_length=1024):
    input_message = [
        {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['input']}.\n{tgt_lang}:"},
    ]
    tokenized_input = tokenizer.apply_chat_template(
        input_message, 
        tokenize=True, 
        add_generation_prompt=True, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True, 
        return_dict=True
    )
    return tokenized_input


# Main function to build datasets
def build_datasets(src_path, tgt_path, src_lang, tgt_lang, tokenizer, max_length=1024, teacher_forcing=True):
    # Wrap with Hugging Face datasets
    dataset = Dataset.from_generator(lambda: line_pairs(f"{os.environ.get('ROOT_DIR')}/{src_path}", f"{os.environ.get('ROOT_DIR')}/{tgt_path}"))

    dataset = dataset.map(
        lambda x: 
            format_and_tokenize_example_for_teacher_forcing(x, src_lang, tgt_lang, tokenizer, max_length) 
            if teacher_forcing
            else format_and_tokenize_example_for_inference(x, src_lang, tgt_lang, tokenizer, max_length),
        load_from_cache_file=True,
        num_proc=100
    )

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + (["labels"] if teacher_forcing else []))

    return dataset
