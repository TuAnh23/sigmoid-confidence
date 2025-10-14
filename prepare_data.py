from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
import os

def line_pairs(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            yield {"input": src.strip(), "target": tgt.strip()}

def _message_spans(messages, tokenizer):
    """
    Returns [(start,end), ...] token spans for each message when the *entire*
    conversation is tokenized with the same chat template settings.
    """
    # same settings used later for the final encoding
    def tok_prefix(n):
        return tokenizer.apply_chat_template(
            messages[:n], tokenize=True, add_generation_prompt=False
        )

    spans = []
    prev_len = 0
    for i in range(1, len(messages) + 1):
        curr_len = len(tok_prefix(i))
        spans.append((prev_len, curr_len))
        prev_len = curr_len
    return spans


def example_to_chat_format(example, dataname, src_lang=None, tgt_lang=None):
    if "google/wmt24pp" in dataname:
        chat_messages = [
            {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['source']}.\n{tgt_lang}: "},
            {"role": "assistant", "content": " " + example['target']}
        ]
    elif dataname == "Unbabel/TowerBlocks-v0.2":
        chat_messages = []
        for turn in example['conversations']:
            role = "user" if turn['from'] == 'human' else "assistant"
            chat_messages.append({
                "role": role, 
                "content": turn['value']
            })
    elif dataname in ["allenai/tulu-3-sft-olmo-2-mixture-0225", "allenai/tulu-3-sft-olmo-2-mixture"]:
        chat_messages = example['messages']
    elif dataname == "pawsx":
        chat_messages = [
            {"role": "user", "content": f"What is a different but equivalent (paraphrase) way of saying: \"{example['input']}\"?\n"},
            {"role": "assistant", "content": " " + example['target']}
        ]
    else:
        chat_messages = [
            {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['input']}.\n{tgt_lang}: "},
            {"role": "assistant", "content": " " + example['target']}
        ]
    
    return chat_messages

# Format for teacher forcing: instruction + response
def format_and_tokenize_example_for_teacher_forcing(example, dataname, src_lang, tgt_lang, tokenizer, max_length=1024):
    full_messages = example_to_chat_format(example, dataname, src_lang, tgt_lang)

    # Compute exact token spans for each message (no pad/trunc here)
    spans = _message_spans(full_messages, tokenizer)

    # Tokenize the full conversation once with padding/truncation
    tokenized_full_prompt = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False, max_length=max_length, 
        padding="max_length", truncation=True, return_dict=True
    )

    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

    # Mask out user prompt in the labels so we dont train on it later

    for (start, end), msg in zip(spans, full_messages):
        if start >= max_length:
            # No need to consider this, as this message is out of context length anyway
            break
        if msg["role"] in ("user", "system"):
            # Do not train on user prompt or system prompt
            mask_start = start
            mask_end = min(end, max_length)
        else:
            # Do not train on generation "bos" prompt, since it will be included in the input during inference
            # (We set add_generation_prompt=True when tokenize the input during inference)
            generation_prompt = tokenizer.apply_chat_template([{"role": "user", "content": ""},], add_generation_prompt=True, tokenize=True)[
                len(tokenizer.apply_chat_template([{"role": "user", "content": ""},], add_generation_prompt=False, tokenize=True)):
            ]  # Subtract the prompt with and without generation prompt to get the generation prompt
            generation_prompt_end = start + len(generation_prompt)
            mask_start = start
            mask_end = min(generation_prompt_end, max_length)

        tokenized_full_prompt["labels"][mask_start:mask_end] = [tokenizer.pad_token_id] * (mask_end - mask_start)

    return tokenized_full_prompt


# Format for batched inference: instruction only
def format_and_tokenize_example_for_inference(example, dataname, src_lang, tgt_lang, tokenizer, max_length=1024):
    full_messages = example_to_chat_format(example, dataname, src_lang, tgt_lang)
    input_message = full_messages[:-1]  # Exclude the last message, which we assume is the gold target
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

def format_raw_strings(example, dataname):
    if "google/wmt24pp" in dataname:
        formatted_example = {
            'src': example['source'],
            'ref': example['target']
        }
    else:
        formatted_example = {
            'src': example['input'],
            'ref': example['target']
        }
    
    return formatted_example


def has_no_none_values(example):
    for turn in example["conversations"]:
        if turn.get("value") is None:
            return False
    return True


# Main function to build datasets
def build_datasets(
        dataname, tokenizer=None, max_length=1024, teacher_forcing=True, # Args used by all datasets
        src_path=None, tgt_path=None, src_lang=None, tgt_lang=None, # Args used by self-loaded data
        split=None, # Args used by huggingface dataset
        raw_text_string=False,  # Return raw text strings with (src, ref) entries, instead of formatted and tokenized input samples
    ):
    if dataname == "Unbabel/TowerBlocks-v0.2":
        dataset = load_dataset(dataname)
        dataset = dataset['train']

        # Extract 5000 samples from the "general_mt_clean" for validation
        general_mt_clean_rows = dataset.filter(lambda x: x["dataset"] == "general_mt_clean")
        general_mt_clean_rows = general_mt_clean_rows.shuffle(seed=123)
        sample_5000 = general_mt_clean_rows.select(range(5000))
        leftover_general_mt_clean = general_mt_clean_rows.select(range(5000, len(general_mt_clean_rows)))
        non_general_mt_clean = dataset.filter(lambda x: x["dataset"] != "general_mt_clean")
        rest = concatenate_datasets([leftover_general_mt_clean, non_general_mt_clean])

        if split == "dev":
            dataset = sample_5000
        else:
            dataset = rest
        dataset = dataset.filter(has_no_none_values)
    elif dataname in ["allenai/tulu-3-sft-olmo-2-mixture-0225", "allenai/tulu-3-sft-olmo-2-mixture"]:
        dataset = load_dataset(dataname)
        dataset = dataset['train']

        # Randomly select 5000 samples for validation
        split_dataset = dataset.train_test_split(test_size=5000, seed=42)
        validation = split_dataset['test']

        # The rest is for training
        train = split_dataset['train']

        if split == "dev":
            dataset = validation
        else:
            dataset = train

    elif "google/wmt24pp" in dataname:
        data_repo, lang_pairs = dataname.split('|')
        dataset = load_dataset(data_repo, lang_pairs)
        dataset = dataset['train'].filter(lambda x: not x["is_bad_source"])
    else:
        # Wrap with Hugging Face datasets
        dataset = Dataset.from_generator(lambda: line_pairs(f"{os.environ.get('ROOT_DIR')}/{src_path}", f"{os.environ.get('ROOT_DIR')}/{tgt_path}"))
    
    if not raw_text_string:
        dataset = dataset.map(
            lambda x: 
                format_and_tokenize_example_for_teacher_forcing(x, dataname, src_lang, tgt_lang, tokenizer, max_length) 
                if teacher_forcing
                else format_and_tokenize_example_for_inference(x, dataname, src_lang, tgt_lang, tokenizer, max_length),
            load_from_cache_file=True,
            num_proc=100
        )

        # Set format for PyTorch
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + (["labels"] if teacher_forcing else []))
    else:
        dataset = dataset.map(
            lambda x: format_raw_strings(x, dataname),
            load_from_cache_file=True,
            num_proc=100,
            remove_columns=dataset.column_names
        )

    return dataset
