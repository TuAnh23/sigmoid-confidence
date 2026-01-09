import torch
import os
import yaml
import re
import json
import itertools

def get_best_checkpoint(output_dir: str, metric: str = "eval_loss", maximize: bool = False):
    # Get all checkpoint directories
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r"checkpoint-\d+", d)]
    if not checkpoints:
        raise ValueError("No checkpoint directories found in output_dir.")

    # Sort checkpoints by step
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

    # Use trainer_state.json from the *last* checkpoint
    last_ckpt_path = os.path.join(output_dir, checkpoints[-1])
    trainer_state_path = os.path.join(last_ckpt_path, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found at {trainer_state_path}")

    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    eval_logs = [log for log in state.get("log_history", []) if metric in log]
    if not eval_logs:
        raise ValueError(f"No evaluation logs found with metric '{metric}'")

    # Find best step
    best_log = max(eval_logs, key=lambda x: x[metric]) if maximize else min(eval_logs, key=lambda x: x[metric])
    best_step = best_log["step"]

    best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{best_step}")
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint directory not found: {best_checkpoint_path}")
    
    print(f"Best checkpoint found at {best_checkpoint_path}.")

    return best_checkpoint_path


def load_yaml_files(yaml_file_paths):
    contents = {}
    for yaml_file_path in yaml_file_paths:
        with open(yaml_file_path, 'r') as file:
            content_part = yaml.safe_load(file)
            if content_part:
                contents.update(content_part)
    return contents

def find_eos_idx(pred_ids, eos_id):
    """
    Find end-of-sentence index to ignore the padding tokens
    """
    match_eos = (pred_ids == eos_id).nonzero(as_tuple=True)[0]
    if match_eos.shape[0] > 0:
        return match_eos[0]  # First time eos
    else:
        return pred_ids.shape[0]
    
def find_start_idx(out_ids, generation_prompt):
    """
    Find the index right after the LAST occurrence of the generation prompt subsequence.
    
    Args:
        out_ids (torch.Tensor): 1D tensor of predicted token ids.
        generation_prompt (list[int]): list of token ids representing the prompt.
    
    Returns:
        int: start index (position after the last occurrence of the prompt).
             Defaults to 0 if not found.
    """
    prompt_len = len(generation_prompt)

    # Walk backward to find the last match
    for i in range(out_ids.shape[0] - prompt_len, -1, -1):
        if torch.equal(out_ids[i:i+prompt_len], torch.tensor(generation_prompt, device=out_ids.device)):
            return i + prompt_len  # start right after the prompt

    # Prompt not found → fall back to 0
    return 0

    
def load_text_file(file_path):
    """
    Load text file into a list, each item is a line of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_comet_model(comet_model_root_path=None, model_name="wmt22-cometkiwi-da"):
    from comet import download_model, load_from_checkpoint

    if model_name in ["wmt22-cometkiwi-da", "wmt20-comet-qe-da", "wmt22-comet-da", "wmt23-cometkiwi-da-xl", "XCOMET-XL", "XCOMET-XXL"]:
        model_path = download_model(f"Unbabel/{model_name}")
    else:
        model_path = f"{comet_model_root_path}/{model_name}/checkpoints/model.ckpt"
    model = load_from_checkpoint(model_path)
    return model

def format_for_comet(src, mt, ref=None):
    if ref is not None:
        return [{'src': x, 'mt': y, 'ref': z} for x, y, z in zip(src, mt, ref)]
    else:
        return [{'src': x, 'mt': y} for x, y in zip(src, mt)]

def write_text_file(lines, file_path):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

def rank_preserving_adjust(A, B, eps=1e-3):
    """
    A, B: torch tensors of shape [..., X]
    eps: relative minimum separation
    """
    # Sort by B (descending) along last dimension
    order = torch.argsort(B, dim=-1, descending=True)
    A_sorted = torch.gather(A, -1, order)

    # Create strictly decreasing upper bounds
    # bound[i] = A_sorted[i-1] * (1 - eps)
    shifted = A_sorted[..., :-1] * (1 - eps)
    shifted = torch.cat([
        torch.full_like(A_sorted[..., :1], float('inf')),
        shifted
    ], dim=-1)

    # Enforce monotonic decrease via cumulative minimum
    C_sorted = torch.minimum(
        A_sorted,
        torch.cummin(shifted, dim=-1).values
    )

    # Scatter back to original order
    C = torch.empty_like(C_sorted)
    C.scatter_(-1, order, C_sorted)

    return C
