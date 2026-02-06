from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from collections import defaultdict
import random
import os
import json
from huggingface_hub import snapshot_download
import pickle

def line_pairs(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            yield {"input": src.strip(), "target": tgt.strip()}

def list_pairs(src_list, tgt_list):
    for src, tgt in zip(src_list, tgt_list):
        yield {"input": src, "target": tgt}

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
    elif "pawsx" in dataname:
        chat_messages = [
            {"role": "user", "content": f"What is a different but equivalent (paraphrase) way of saying: \"{example['input']}\"?\n"},
            {"role": "assistant", "content": " " + example['target']}
        ]
    elif "sciex" in dataname:
        chat_messages = [
            {"role": "user", "content": f"You are a university student. Please answer the following JSON-formatted exam question. The subquestions (if any) are indexed. Please give the answers to the question and subquestions that were asked, and index them accordingly in your output. You do not have to provide your output in the JSON format. Here is the question: \n\n{example['input']}"},
            {"role": "assistant", "content": " " + example['target']}
        ]
    elif "truthfulqa" in dataname:
        chat_messages = [
            {"role": "user", "content": example['Question']},
            {"role": "assistant", "content": " " + example['Best Answer']}
        ]
    elif "gsm8k" in dataname:
        chat_messages = [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": " " + example['answer']}
        ]
    elif "xsum" in dataname:
        chat_messages = [
            {"role": "user", "content": f"Summarize the following document in one sentence: \n\n{example['input']}"},
            {"role": "assistant", "content": " " + example['target']}
        ]
    else:
        chat_messages = [
            {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['input']}.\n{tgt_lang}: "},
            {"role": "assistant", "content": " " + example['target']}
        ]
    
    return chat_messages


def _get_generation_prompt_tokens(tokenizer):
    """
    Get the generation prompt tokens that are added between user and assistant messages.
    This is computed by subtracting tokenization with and without add_generation_prompt.
    
    Returns:
        List of token ids representing the generation prompt
    """
    with_gen_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""},], 
        add_generation_prompt=True, 
        tokenize=True
    )
    without_gen_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""},], 
        add_generation_prompt=False, 
        tokenize=True
    )
    return with_gen_prompt[len(without_gen_prompt):]


def _mask_labels_for_teacher_forcing(tokenized_prompt, spans, messages, tokenizer, max_length):
    """
    Mask out labels for positions we don't want to train on (user/system prompts and generation prompts).
    
    This function modifies tokenized_prompt["labels"] in place.
    
    Args:
        tokenized_prompt: Dict with "labels" key to be modified
        spans: List of (start, end) token spans for each message
        messages: List of chat message dicts with "role" and "content" keys
        tokenizer: The tokenizer (needed for pad_token_id and generation prompt)
        max_length: Maximum sequence length
        
    Returns:
        List of (content_start, content_end) tuples for assistant message content regions
        (useful for MQM label mapping)
    """
    generation_prompt = _get_generation_prompt_tokens(tokenizer)
    assistant_content_spans = []
    
    for (start, end), msg in zip(spans, messages):
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
            generation_prompt_end = start + len(generation_prompt)
            mask_start = start
            mask_end = min(generation_prompt_end, max_length)
            
            # Record the content span (after generation prompt) for potential MQM label mapping
            content_start = generation_prompt_end
            content_end = min(end, max_length)
            assistant_content_spans.append((content_start, content_end))
        
        tokenized_prompt["labels"][mask_start:mask_end] = [tokenizer.pad_token_id] * (mask_end - mask_start)
    
    return assistant_content_spans


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

    # Mask out user prompt and generation prompts in the labels so we don't train on them
    _mask_labels_for_teacher_forcing(tokenized_full_prompt, spans, full_messages, tokenizer, max_length)

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

def format_raw_data(example, dataname):
    """
    Return raw text (untokenized)
    In the case of MQM/ESA: return character-level span instead of token level labels
    """
    if "google/wmt24pp" in dataname:
        formatted_example = {
            'src': example['source'],
            'ref': example['target']
        }
    elif "truthfulqa" in dataname:
        formatted_example = {
            'src': example['Question'],
            'ref': example['Best Answer'],
            'correct_answers': example['Correct Answers'],
            'incorrect_answers': example['Incorrect Answers'],
        }
    elif "gsm8k" in dataname:
        formatted_example = {
            'src': example['question'],
            'ref': example['answer']
        }
    elif dataname == "RicardoRei/wmt-mqm-error-spans" or dataname.startswith("wmt24_esa"):
        formatted_example = {
            'src': example['src'],
            'mt': example['mt'],
            'annotations': example['annotations'],
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


# ==================== MQM Dataset Preprocessing Functions ====================

def load_wmt24_esa_dataset(jsonl_path, lang_pair=None):
    """
    Load WMT24 ESA dataset from a JSONL file and convert to MQM format.
    
    Args:
        jsonl_path: Path to the JSONL file
        lang_pair: Optional language pair filter (e.g., "en-cs"). If None, load all.
        
    Returns:
        HuggingFace Dataset with fields: src, mt, lp, annotations
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Filter by language pair if specified
            if lang_pair is not None and item['langs'] != lang_pair:
                continue
            
            # Convert esa_spans to annotations format
            # esa_spans uses start_i/end_i (character indices), we convert to start/end
            annotations = []
            for span in item.get('esa_spans', []):
                if span['start_i'] == 'missing' or span['end_i'] == 'missing':
                    continue
                annotations.append({
                    'start': span['start_i'],
                    'end': span['end_i'] + 1,  # Make end exclusive to match MQM format
                    'severity': span.get('severity', ''),
                    'text': item['tgt'][span['start_i']:span['end_i'] + 1] if span['start_i'] < len(item['tgt']) else ''
                })
            
            data.append({
                'src': item['src'],
                'mt': item['tgt'],
                'lp': item['langs'],
                'annotations': annotations,
            })
    
    return Dataset.from_list(data)


def mqm_deduplicate_dataset(dataset, mode):
    """
    Deduplicate MQM dataset based on the specified mode.
    
    Args:
        dataset: HuggingFace dataset with MQM samples
        mode: Deduplication mode:
            - "src": Keep only one sample per unique source sentence
            - "src_mt": Keep only one sample per unique (source, MT) pair
            - "src_mt_annotations": Keep only one sample per unique (source, MT, annotations) triple
            
    Returns:
        Deduplicated dataset
    """
    if mode is None:
        return dataset
    
    print(f"MQM deduplication mode: {mode}")
    original_size = len(dataset)
    
    if mode == "src":
        # Keep only one sample per unique source sentence
        seen_sources = set()
        keep_indices = []
        for i, example in enumerate(dataset):
            if example['src'] not in seen_sources:
                seen_sources.add(example['src'])
                keep_indices.append(i)
        dataset = dataset.select(keep_indices)
        
    elif mode == "src_mt":
        # Keep only one sample per unique (source, MT) pair
        seen_pairs = set()
        keep_indices = []
        for i, example in enumerate(dataset):
            key = (example['src'], example['mt'])
            if key not in seen_pairs:
                seen_pairs.add(key)
                keep_indices.append(i)
        dataset = dataset.select(keep_indices)
        
    elif mode == "src_mt_annotations":
        # Keep only one sample per unique (source, MT, annotations) triple
        # Annotations are converted to a hashable format (tuple of tuples)
        seen_triples = set()
        keep_indices = []
        for i, example in enumerate(dataset):
            annotations = example.get('annotations', [])
            # Convert annotations to hashable format
            annotations_tuple = tuple(
                (a['start'], a['end'], a.get('severity', ''), a.get('text', ''))
                for a in sorted(annotations, key=lambda x: (x['start'], x['end']))
            ) if annotations else ()
            key = (example['src'], example['mt'], annotations_tuple)
            if key not in seen_triples:
                seen_triples.add(key)
                keep_indices.append(i)
        dataset = dataset.select(keep_indices)
    else:
        raise ValueError(f"Unknown mqm_deduplicate mode: {mode}. "
                        f"Valid options: 'src', 'src_mt', 'src_mt_annotations'")
    
    print(f"  Deduplication: {original_size} -> {len(dataset)} samples "
          f"({100 * len(dataset) / original_size:.1f}% retained)")
    
    return dataset


def mqm_source_aware_split(dataset, split, dev_size=5000):
    """
    Split MQM dataset into train/dev ensuring samples with the same source 
    don't appear in both sets (prevents data leakage).
    
    Args:
        dataset: HuggingFace dataset with MQM samples
        split: "train" or "dev"
        dev_size: Target number of samples for dev set (approximate, due to source-level split)
        
    Returns:
        Dataset split (either train or dev portion)
    """
    
    # Group samples by source sentence
    src_to_indices = defaultdict(list)
    for i, example in enumerate(dataset):
        src_to_indices[example['src']].append(i)
    
    # Shuffle unique sources and split them
    random.seed(42)
    unique_sources = list(src_to_indices.keys())
    random.shuffle(unique_sources)
    
    # Calculate split point (aim for ~dev_size dev samples, but ensure source-level split)
    cumulative_samples = 0
    dev_source_count = 0
    for src in unique_sources:
        cumulative_samples += len(src_to_indices[src])
        dev_source_count += 1
        if cumulative_samples >= dev_size:
            break
    
    # Split sources into dev and train
    dev_sources = set(unique_sources[:dev_source_count])
    train_sources = set(unique_sources[dev_source_count:])
    
    # Collect indices for each split
    dev_indices = []
    train_indices = []
    for src, indices in src_to_indices.items():
        if src in dev_sources:
            dev_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    print(f"MQM source-aware split:")
    print(f"  Unique sources: {len(unique_sources)}")
    print(f"  Dev sources: {len(dev_sources)} -> {len(dev_indices)} samples")
    print(f"  Train sources: {len(train_sources)} -> {len(train_indices)} samples")
    
    if split == "dev":
        return dataset.select(dev_indices)
    else:
        return dataset.select(train_indices)


# Main function to build datasets
def build_datasets(
        dataname, tokenizer=None, max_length=1024, teacher_forcing=True, # Args used by all datasets
        src_path=None, tgt_path=None, src_lang=None, tgt_lang=None, # Args used by self-loaded data
        split=None, # Args used by huggingface dataset. Use the full dataset if set to None
        raw_text_string=False,  # Return raw text strings with (src, ref) entries, instead of formatted and tokenized input samples
        mqm_deduplicate=None,  # MQM deduplication mode: None, "src", "src_mt", "src_mt_annotations"
        mqm_filter_no_annotations=False,  # If True, filter out MQM samples without error annotations
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
    elif dataname == "RicardoRei/wmt-mqm-error-spans":
        # ==================== MQM Error Spans Dataset ====================
        # This dataset contains machine translation outputs with error span annotations.
        # We use it for token-level confidence training where erroneous tokens are treated
        # as negative examples only (no additional negative sampling).
        dataset = load_dataset(dataname, split="train")
        
        # Filter out samples without annotations if requested
        if mqm_filter_no_annotations:
            dataset = dataset.filter(lambda x: x.get('annotations') and len(x['annotations']) > 0)
        
        # Apply deduplication if requested
        # Options: "src" (unique sources), "src_mt" (unique src+mt pairs), 
        #          "src_mt_annotations" (unique src+mt+annotations)
        dataset = mqm_deduplicate_dataset(dataset, mode=mqm_deduplicate)
        
        if split is not None:
            # Source-aware train/dev split
            # Ensure samples with the same source don't appear in both train and dev sets
            dataset = mqm_source_aware_split(dataset, split=split, dev_size=5000)
    elif dataname.startswith("wmt24_esa"):
        # ==================== WMT24 ESA Dataset ====================
        # Local JSONL dataset with ESA (Error Span Annotation) format.
        # Converted to MQM format for token-level confidence training.
        jsonl_path = f"{os.environ.get('ROOT_DIR')}/{src_path}"
        
        # Parse language pair from tgt_lang if provided (reusing tgt_lang param for lp)
        lang_pair = tgt_lang  # e.g., "en-cs"
        
        dataset = load_wmt24_esa_dataset(jsonl_path, lang_pair=lang_pair)
        
        # Filter out samples without annotations if requested
        if mqm_filter_no_annotations:
            dataset = dataset.filter(lambda x: x.get('annotations') and len(x['annotations']) > 0)
        
        # Apply deduplication if requested
        dataset = mqm_deduplicate_dataset(dataset, mode=mqm_deduplicate)
        
        if split is not None:
            # Source-aware train/dev split
            dataset = mqm_source_aware_split(dataset, split=split, dev_size=5000)

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
    elif "sciex" in dataname:
        # Wrap with Hugging Face datasets
        src_list, tgt_list = load_sciex()
        dataset = Dataset.from_generator(lambda: list_pairs(src_list, tgt_list))
    elif "truthfulqa" in dataname:
        dataset = load_dataset("domenicrosati/TruthfulQA")['train']
    elif "gsm8k" in dataname:
        dataset = load_dataset("openai/gsm8k", 'main')['test']
    elif "xsum" in dataname:
        with open(f"{os.environ.get('ROOT_DIR')}/{src_path}", "rb") as f:
            src_list = pickle.load(f)
        with open(f"{os.environ.get('ROOT_DIR')}/{tgt_path}", "rb") as f:
            tgt_list = pickle.load(f)
        dataset = Dataset.from_generator(lambda: list_pairs(src_list, tgt_list))
    else:
        # Wrap with Hugging Face datasets
        dataset = Dataset.from_generator(lambda: line_pairs(f"{os.environ.get('ROOT_DIR')}/{src_path}", f"{os.environ.get('ROOT_DIR')}/{tgt_path}"))
    
    if not raw_text_string:
        # Use special tokenization for MQM data that creates token-level labels
        if dataname == "RicardoRei/wmt-mqm-error-spans" or dataname.startswith("wmt24_esa"):
            dataset = dataset.map(
                lambda x: format_and_tokenize_mqm_example_for_teacher_forcing(x, tokenizer, max_length),
                load_from_cache_file=False,
                num_proc=100
            )
            # Set format for PyTorch - include mqm_token_labels for MQM training mode
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "mqm_token_labels"])
        else:
            dataset = dataset.map(
                lambda x: 
                    format_and_tokenize_example_for_teacher_forcing(x, dataname, src_lang, tgt_lang, tokenizer, max_length) 
                    if teacher_forcing
                    else format_and_tokenize_example_for_inference(x, dataname, src_lang, tgt_lang, tokenizer, max_length),
                load_from_cache_file=False,
                num_proc=100
            )

            # Set format for PyTorch
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + (["labels"] if teacher_forcing else []))
    else:
        dataset = dataset.map(
            lambda x: format_raw_data(x, dataname),
            load_from_cache_file=False,
            num_proc=100,
            remove_columns=dataset.column_names
        )

    return dataset


def load_sciex(
        kept_lang="en"  # "all", "en", "de"
    ):
    """
    Loads the SciEx dataset from Huggingface hub.
    """
    # Define helper functions
    def info_from_exam_path(exam_json_path):
        exam_name = exam_json_path.split('/')[-2]
        lang = exam_json_path.split('/')[-1].replace('.json', '').split('_')[-1]
        assert lang in ['en', 'de']
        return exam_name, lang
    
    def load_json(file_path):
        with open(file_path, 'r') as f:
            obj = json.load(f)
        return obj
    
    def is_none(x):
        if (x is None) or (isinstance(x, str) and x.lower() == "none"):
            return True
        return False
    
    def return_gold_answer(question, lang):
        if lang == 'en':
            same_lang_answer = question['GoldAnswerEnglish'] if not is_none(question['GoldAnswerEnglish']) else None
            diff_lang_answer = question['GoldAnswerGerman'] if not is_none(question['GoldAnswerGerman']) else None
        elif lang == 'de':
            same_lang_answer = question['GoldAnswerGerman'] if not is_none(question['GoldAnswerGerman']) else None
            diff_lang_answer = question['GoldAnswerEnglish'] if not is_none(question['GoldAnswerEnglish']) else None
        else:
            raise RuntimeError(f"Invalid lang `{lang}`")

        if same_lang_answer is not None:
            return '\n\n'.join(same_lang_answer) if isinstance(same_lang_answer, list) else same_lang_answer
        elif diff_lang_answer is not None:
            return '\n\n'.join(diff_lang_answer) if isinstance(diff_lang_answer, list) else diff_lang_answer
        else:
            return None
        
    def collect_figures(question_dict):
        figure_list = []
        if 'Figures' in question_dict:
            figure_list.extend(question_dict['Figures'])
        if 'Subquestions' in question_dict:
            for subquestion_dict in question_dict['Subquestions']:
                if 'Figures' in subquestion_dict:
                    figure_list.extend(subquestion_dict['Figures'])
        return figure_list

    # Download the entire dataset repo
    local_dir = snapshot_download(
        repo_id="tuanh23/SciEx",
        repo_type="dataset"
    )

    question_list = []
    gold_answer_list = []

    # Loop though all the files ending with .json
    exam_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(f"{local_dir}/exams_json")
        for f in files if f.endswith('.json')
    ]
    for exam_file in exam_files:
        exam_name, lang = info_from_exam_path(exam_file)
        if kept_lang != "all" and lang != kept_lang:
            continue
        exam = load_json(exam_file)
        # Load gold solutions
        solution_file = f"{local_dir}/human_feedback/{exam_name}/additional_info.json"
        if not os.path.isfile(solution_file):
            continue
        solution = load_json(solution_file)
        for qid in range(len(exam['Questions'])):
            exam['Questions'][qid].pop("Index")
            question = exam['Questions'][qid]
            question_solution = return_gold_answer(solution['Questions'][qid], lang)

            if question_solution is not None and len(collect_figures(question)) == 0:
                question_list.append(json.dumps(exam['Questions'][qid]))
                gold_answer_list.append(question_solution)

    return question_list, gold_answer_list


# ==================== MQM (Machine Translation Quality Metrics) Data Utilities ====================
# These functions handle the MQM error span data for token-level confidence training.
# MQM data contains error annotations in the MT output, which we use to create per-token labels:
#   - Label = 1: token is correct (not in any error span)
#   - Label = 0: token is erroneous (overlaps with an error span)
# During training:
#   - Correct tokens (label=1): train as positive with negative sampling (standard approach)
#   - Erroneous tokens (label=0): train as single negative (no additional negative sampling)

def get_char_to_token_mapping(text, tokenizer):
    """
    Create a mapping from character positions to token indices.
    
    Args:
        text: The text string to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        char_to_token: List where char_to_token[i] is the token index for character i
    """
    # Tokenize with offset mapping to get character spans for each token
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    
    # Create character-to-token mapping
    char_to_token = [-1] * len(text)  # -1 means not mapped to any token
    for token_idx, (start, end) in enumerate(offsets):
        for char_idx in range(start, end):
            if char_idx < len(text):
                char_to_token[char_idx] = token_idx
    
    return char_to_token


def create_token_labels_from_mqm_annotations(mt_text, annotations, tokenizer):
    """
    Create per-token binary labels based on MQM error span annotations.
    
    Args:
        mt_text: The machine translation text
        annotations: List of annotation dicts with 'start', 'end', 'severity', 'text' keys
        tokenizer: The tokenizer to use
        
    Returns:
        token_labels: List of 0/1 labels for each token (1 = correct, 0 = erroneous)
    """
    # Get tokenization
    encoding = tokenizer(mt_text, add_special_tokens=False)
    num_tokens = len(encoding['input_ids'])
    
    # Initialize all tokens as correct (label = 1)
    token_labels = [1] * num_tokens
    
    if not annotations:
        return token_labels
    
    # Get character to token mapping
    char_to_token = get_char_to_token_mapping(mt_text, tokenizer)
    
    # Mark tokens that overlap with any error span as erroneous (label = 0)
    for annotation in annotations:
        start_char = annotation['start']
        end_char = annotation['end']
        
        # Find all tokens that overlap with this error span
        for char_idx in range(start_char, min(end_char, len(char_to_token))):
            token_idx = char_to_token[char_idx]
            if token_idx >= 0 and token_idx < num_tokens:
                token_labels[token_idx] = 0
    
    return token_labels


def get_lang_map():
    return {
        'en': 'English',
        'de': 'German',
        'zh': 'Chinese',
        'ru': 'Russian',
        'is': 'Icelandic',
        'hi': 'Hindi',
        'ja': 'Japanese',
        'es': 'Spanish',
        'uk': 'Ukrainian',
        'cs': 'Czech',
    }


def format_and_tokenize_mqm_example_for_teacher_forcing(example, tokenizer, max_length=1024):
    """
    Format and tokenize an MQM example for teacher forcing training.
    Creates token-level labels based on error span annotations.
    
    Args:
        example: Dict with 'src', 'mt', 'ref', 'annotations', 'lp' (language pair) keys
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dict with input_ids, attention_mask, labels, and mqm_token_labels
    """
    # Parse language pair (e.g., "en-de" -> "English", "German")
    lp = example['lp']
    src_code, tgt_code = lp.split('-')
    lang_map = get_lang_map()
    src_lang = lang_map[src_code]
    tgt_lang = lang_map[tgt_code]
    
    # Create chat messages using MT output (not reference) as target
    # This is because we want to train the model to recognize errors in MT output
    chat_messages = [
        {"role": "user", "content": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {example['src']}.\n{tgt_lang}: "},
        {"role": "assistant", "content": " " + example['mt']}
    ]
    
    # Compute exact token spans for each message
    spans = _message_spans(chat_messages, tokenizer)
    
    # Tokenize the full conversation
    tokenized_full_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=True, add_generation_prompt=False, max_length=max_length, 
        padding="max_length", truncation=True, return_dict=True
    )
    
    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
    
    # Create MQM token labels for the MT output portion
    # First, get labels for the raw MT text
    mt_token_labels = create_token_labels_from_mqm_annotations(
        example['mt'], 
        example.get('annotations', []), 
        tokenizer
    )
    
    # Initialize mqm_token_labels with -1 (ignore) for the full sequence
    # -1 means this position should be ignored for MQM-specific training logic
    mqm_token_labels = [-1] * len(tokenized_full_prompt["input_ids"])
    
    # Mask out user prompt and generation prompts in labels, and get assistant content spans
    assistant_content_spans = _mask_labels_for_teacher_forcing(
        tokenized_full_prompt, spans, chat_messages, tokenizer, max_length
    )
    
    # Map MQM token labels to the assistant content regions
    for (content_start, content_end) in assistant_content_spans:
        content_length = content_end - content_start
        # Assign MQM labels to the content tokens
        for i in range(min(content_length, len(mt_token_labels))):
            if content_start + i < max_length:
                mqm_token_labels[content_start + i] = mt_token_labels[i]
    
    tokenized_full_prompt["mqm_token_labels"] = mqm_token_labels
    
    return tokenized_full_prompt