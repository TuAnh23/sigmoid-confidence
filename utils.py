import torch
import os
import wandb
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


def find_dominant(softmax_lprobs, find_dominant_method, p_jump=None, diff_cut=None):
    """
    Return binary tensor shape [batch_size, nr_tokens, vocab_size], where 1 indicates that the probability
    value at that position is dominant, 0 otherwise
    """
    p_threshold = float(os.environ.get('P_THRESHOLD')) if os.environ.get('P_THRESHOLD') is not None else None
    epsilon = float(os.environ.get('EPSILON')) if os.environ.get('EPSILON') is not None else None
    k = int(os.environ.get('K')) if os.environ.get('K') is not None else None
    top_p = float(os.environ.get('TOP_P')) if os.environ.get('TOP_P') is not None else None
    min_p = float(os.environ.get('MIN_P')) if os.environ.get('MIN_P') is not None else None
    entropy_portion_cut = float(
        os.environ.get('ENTROPY_PORTION_CUT')
    ) if os.environ.get('ENTROPY_PORTION_CUT') is not None else None

    prob_dist = torch.exp(softmax_lprobs)
    sorted_prob_dist, indices = torch.sort(prob_dist, descending=True, dim=-1)
    if find_dominant_method == "prob_threshold":
        # assert p_threshold is not None
        # p_threshold = 0.2
        wandb.config['p_threshold'] = p_threshold
        mask = sorted_prob_dist > p_threshold
    elif find_dominant_method == "eta":
        # epsilon = 0.09
        wandb.config['epsilon'] = epsilon
        epsilon = torch.tensor(epsilon)
        entropy = -torch.mul(softmax_lprobs, torch.exp(softmax_lprobs)).sum(dim=-1)
        entropy = entropy.unsqueeze(-1).expand_as(softmax_lprobs)
        mask = (sorted_prob_dist > epsilon) | (sorted_prob_dist > torch.sqrt(epsilon) * torch.exp(-entropy))
    elif find_dominant_method == "top-k":
        # k = 5
        wandb.config['k'] = k
        mask = torch.zeros_like(sorted_prob_dist, dtype=torch.bool)
        mask[...,:k] = True
    elif find_dominant_method == "top-p":
        # top_p = 0.8
        wandb.config['top_p'] = top_p
        cumulative_sum = torch.cumsum(sorted_prob_dist, dim=-1)
        mask = cumulative_sum < top_p
    elif find_dominant_method == "min-p":
        # min_p = 0.1
        wandb.config['min_p'] = min_p
        mask = sorted_prob_dist > min_p * sorted_prob_dist[..., 0].unsqueeze(-1).expand_as(sorted_prob_dist)
    elif find_dominant_method == "difference_jump":
        assert p_jump is not None
        diff = sorted_prob_dist[..., :-1] - sorted_prob_dist[..., 1:]
        # Identify the cutoff condition along the last dimension
        # mask = diff[..., :-1] > 4 * diff[..., 1:]  # Shape: [batch_size, nr_tokens, vocab_size - 2]
        mask = (diff > p_jump * sorted_prob_dist[..., :-1]) & (diff > diff_cut)
    elif find_dominant_method == "difference_jump_entropy_cut":
        wandb.config["p_jump"] = p_jump
        wandb.config['entropy_portion_cut'] = entropy_portion_cut

        sorted_prob_dist_ascending = torch.flip(sorted_prob_dist, dims=[-1])
        normalize_dennominator = torch.cumsum(sorted_prob_dist_ascending, dim=-1)
        leave_out_entropies = - (torch.cumsum(torch.mul(sorted_prob_dist_ascending, torch.log(sorted_prob_dist_ascending)), dim=-1) / normalize_dennominator) \
                              + torch.cumsum(sorted_prob_dist_ascending, dim=-1) * torch.log(normalize_dennominator) / normalize_dennominator
        leave_out_entropies = torch.flip(leave_out_entropies, dims=[-1])
        vocab_size = torch.tensor(sorted_prob_dist.shape[-1], device=sorted_prob_dist.device)

        assert p_jump is not None
        diff = sorted_prob_dist[..., :-1] - sorted_prob_dist[..., 1:]
        # Identify the cutoff condition along the last dimension
        # mask = diff[..., :-1] > 4 * diff[..., 1:]  # Shape: [batch_size, nr_tokens, vocab_size - 2]
        mask = (diff > p_jump * sorted_prob_dist[..., :-1]) & \
               (leave_out_entropies[..., :-1] < entropy_portion_cut * torch.log(vocab_size)) & \
               (sorted_prob_dist[..., :-1] > 1 / vocab_size)
    else:
        raise RuntimeError(f"Unknown find_dominant_method {find_dominant_method}")

    # Get the last occurrence of True along the last axis
    cut_points = mask.shape[-1] - 1 - torch.argmax(torch.flip(mask, dims=[-1]).int(),
                                                   dim=-1)  # Shape: [batch_size, nr_tokens]

    # Handle cases where no cutoff is found (all False)
    no_cutoff = ~mask.any(axis=-1)
    cut_points[no_cutoff] = -1  # Use -1 to indicate no valid cutoff found

    # import pickle
    # with open('/project/OML/tdinh/tmp_storage/prob_dist.pkl', 'wb') as f:
    #     pickle.dump(prob_dist, f)
    # with open('/project/OML/tdinh/tmp_storage/cut_points.pkl', 'wb') as f:
    #     pickle.dump(cut_points, f)
    # exit()

    # Assuming `indices` is of shape [batch_size, nr_tokens, vocab_size]
    batch_indices = torch.arange(indices.shape[-1], device=indices.device).expand_as(indices)

    # Ensure cut_point has the same shape as batch_indices (for broadcasting)
    cut_point_expanded = cut_points.unsqueeze(-1)  # Shape: [batch_size, nr_tokens, 1]

    # Create mask: Select elements up to cut_point, but disable selection when cut_point == -1
    mask = (batch_indices <= cut_point_expanded) & (cut_point_expanded != -1)

    # Mask out indices beyond the cut-off point with value -1
    indices = torch.where(mask, indices, -1)
    return indices


def check_is_dominant(softmax_lprobs, label_ids, ue_method="sum_dominant_mass", find_dominant_method='difference_jump', p_jump=0.3, diff_cut=0.005):
    dominant_indices = find_dominant(softmax_lprobs, find_dominant_method, p_jump, diff_cut)  # Shape: [batch_size, nr_tokens, vocab_size]

    # Check if each predicted_id is in the corresponding dominant indices
    is_dominant = (label_ids.unsqueeze(-1) == dominant_indices).any(dim=-1)  # Shape: [batch_size, nr_tokens]

    if ue_method == "is_dominant":
        return is_dominant.int()  # Convert to 0/1 format
    elif ue_method == "sum_dominant_mass":
        trans_probs = torch.exp(softmax_lprobs)
        dominant_binary = torch.zeros_like(trans_probs, dtype=torch.uint8)  # Initialize binary tensor

        lists = [list(range(x)) for x in trans_probs.shape[:-1]]
        if len(lists) > 0:
            combs = tuple(itertools.product(*lists))
            for comb in combs:
                dominant_binary[comb][dominant_indices[comb][dominant_indices[comb] != -1]] = 1
        else:
            dominant_binary[dominant_indices[dominant_indices != -1]] = 1

        dominant_mass = (dominant_binary * trans_probs).sum(dim=-1)
        selected_prob = trans_probs.gather(dim=-1, index=label_ids.unsqueeze(-1)).squeeze(-1)
        sum_dominant_mass = torch.where(is_dominant, dominant_mass, selected_prob)
        return sum_dominant_mass
    else:
        raise RuntimeError(f"Unknown ue_method {ue_method}")

def find_eos_idx(pred_ids, eos_id):
    """
    Find end-of-sentence index to ignore the padding tokens
    """
    match_eos = (pred_ids == eos_id).nonzero(as_tuple=True)[0]
    if match_eos.shape[0] > 0:
        return match_eos[0]  # First time eos
    else:
        return pred_ids.shape[0]