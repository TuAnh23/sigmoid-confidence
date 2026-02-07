import numpy as np
import argparse
import os
import wandb
from transformers import set_seed
from utils import load_yaml_files, load_text_file, load_comet_model, format_for_comet, write_text_file
import json
from scipy.stats import pearsonr, spearmanr, kendalltau
from prepare_data import build_datasets
from sacrebleu.metrics import BLEU, CHRF
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from collections import defaultdict
from transformers import AutoTokenizer


def min_max_scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if np.min(arr) != np.max(arr) else np.nan


def pairwise_accuracy_with_ties(gold_scores, metric_scores, epsilon=0.0):
    """
    Calculate pairwise accuracy (acc23 variant) for a single group.
    
    For each pair of translations:
    - Concordant: metric and gold agree on ranking
    - Discordant: metric and gold disagree
    - tie_gold_only: gold says equal, metric says different
    - tie_metric_only: metric says equal, gold says different
    - tie_both: both say equal
    
    acc23 = (concordant + tie_both) / total_pairs
    
    Args:
        gold_scores: List/array of gold quality scores for this group
        metric_scores: List/array of metric scores for this group  
        epsilon: Threshold for considering metric scores as tied
        
    Returns:
        Tuple of (accuracy, num_pairs)
    """
    n = len(gold_scores)
    if n < 2:
        return np.nan, 0
    
    concordant = 0
    discordant = 0
    tie_gold_only = 0
    tie_metric_only = 0
    tie_both = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            g_i, g_j = gold_scores[i], gold_scores[j]
            m_i, m_j = metric_scores[i], metric_scores[j]
            
            # Skip if any is NaN
            if np.isnan(g_i) or np.isnan(g_j) or np.isnan(m_i) or np.isnan(m_j):
                continue
            
            # Determine gold ranking
            if g_i == g_j:
                gold_tie = True
            else:
                gold_tie = False
                gold_i_better = g_i > g_j
            
            # Determine metric ranking (with epsilon threshold for ties)
            if abs(m_i - m_j) <= epsilon:
                metric_tie = True
            else:
                metric_tie = False
                metric_i_better = m_i > m_j
            
            # Classify the pair
            if gold_tie and metric_tie:
                tie_both += 1
            elif gold_tie and not metric_tie:
                tie_gold_only += 1
            elif not gold_tie and metric_tie:
                tie_metric_only += 1
            elif gold_i_better == metric_i_better:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant + tie_gold_only + tie_metric_only + tie_both
    
    if total_pairs == 0:
        return np.nan, 0
    
    # acc23 formula: gives credit for correctly predicting ties
    accuracy = (concordant + tie_both) / total_pairs
    
    return accuracy, total_pairs


class _RankedPair:
    """Maintains metadata for a ranked pair for calculating Kendall's tau efficiently."""
    
    def __init__(self, g1, g2, m1, m2, item_id):
        self.item_id = item_id
        self.diff = abs(m1 - m2)
        
        # Determine stats when treated normally (no tie introduced)
        gold_tie = (g1 == g2)
        metric_tie = (m1 == m2)
        
        if gold_tie and metric_tie:
            self.con, self.dis, self.t_gold, self.t_metric, self.t_both = 0, 0, 0, 0, 1
        elif gold_tie:
            self.con, self.dis, self.t_gold, self.t_metric, self.t_both = 0, 0, 1, 0, 0
        elif metric_tie:
            self.con, self.dis, self.t_gold, self.t_metric, self.t_both = 0, 0, 0, 1, 0
        elif (g1 > g2 and m1 > m2) or (g1 < g2 and m1 < m2):
            self.con, self.dis, self.t_gold, self.t_metric, self.t_both = 1, 0, 0, 0, 0
        else:
            self.con, self.dis, self.t_gold, self.t_metric, self.t_both = 0, 1, 0, 0, 0
        
        # Determine stats when a tie is introduced in metric (epsilon >= diff)
        if gold_tie:
            self.tie_con, self.tie_dis, self.tie_t_gold, self.tie_t_metric, self.tie_t_both = 0, 0, 0, 0, 1
        else:
            self.tie_con, self.tie_dis, self.tie_t_gold, self.tie_t_metric, self.tie_t_both = 0, 0, 0, 1, 0


def find_optimal_epsilon(gold_scores_by_item, metric_scores_by_item, sample_rate=1.0):
    """
    Find the optimal epsilon threshold that maximizes group-by-item accuracy.
    
    Uses an efficient incremental algorithm: sort pairs by metric difference,
    then sweep through thresholds updating statistics incrementally.
    
    Complexity: O(P log P) where P is total number of pairs, instead of O(T * P).
    
    Args:
        gold_scores_by_item: Dict mapping item_id -> list of gold scores
        metric_scores_by_item: Dict mapping item_id -> list of metric scores
        sample_rate: Proportion of pairs to sample (1.0 = all pairs)
        
    Returns:
        Tuple of (best_epsilon, best_accuracy)
    """
    # Enumerate all pairs with their statistics
    pairs = []
    item_ids = set()
    
    for item_id in gold_scores_by_item:
        gold = gold_scores_by_item[item_id]
        metric = metric_scores_by_item[item_id]
        n = len(gold)
        
        for i in range(n):
            for j in range(i + 1, n):
                g_i, g_j = gold[i], gold[j]
                m_i, m_j = metric[i], metric[j]
                
                # Skip NaN pairs
                if np.isnan(g_i) or np.isnan(g_j) or np.isnan(m_i) or np.isnan(m_j):
                    continue
                
                # Sample pairs if sample_rate < 1
                if sample_rate < 1.0 and np.random.random() > sample_rate:
                    continue
                
                pairs.append(_RankedPair(g_i, g_j, m_i, m_j, item_id))
                item_ids.add(item_id)
    
    if not pairs:
        return 0.0, np.nan
    
    num_items = len(item_ids)
    
    # Initialize per-item statistics (concordant + ties_both for acc23)
    item_stats = {item_id: {'con': 0, 't_both': 0, 'total': 0} for item_id in item_ids}
    
    for pair in pairs:
        item_stats[pair.item_id]['con'] += pair.con
        item_stats[pair.item_id]['t_both'] += pair.t_both
        item_stats[pair.item_id]['total'] += 1
    
    # Calculate initial accuracy (epsilon = 0)
    def calc_avg_accuracy():
        accs = []
        for item_id in item_ids:
            stats = item_stats[item_id]
            if stats['total'] > 0:
                accs.append((stats['con'] + stats['t_both']) / stats['total'])
        return np.mean(accs) if accs else np.nan
    
    thresholds = [0.0]
    taus = [calc_avg_accuracy()]
    
    # Sort pairs by their metric difference
    pairs.sort(key=lambda p: p.diff)
    
    # Sweep through thresholds, incrementally updating statistics
    for pair in pairs:
        # Remove old stats, add tie stats (as if epsilon >= pair.diff)
        item_stats[pair.item_id]['con'] -= pair.con
        item_stats[pair.item_id]['con'] += pair.tie_con
        item_stats[pair.item_id]['t_both'] -= pair.t_both
        item_stats[pair.item_id]['t_both'] += pair.tie_t_both
        
        avg_acc = calc_avg_accuracy()
        
        # If same threshold as last, update; otherwise add new
        if thresholds[-1] == pair.diff:
            taus[-1] = avg_acc
        else:
            thresholds.append(pair.diff)
            taus.append(avg_acc)
    
    # Find best
    best_idx = np.nanargmax(taus)
    return thresholds[best_idx], taus[best_idx]


def groupby_item_accuracy_with_tie_calibration(
    src_sentences, 
    gold_scores, 
    metric_scores, 
    sample_rate=1.0,
    return_details=False
):
    """
    Calculate group-by-item segment-level accuracy with tie calibration (acc*eq).
    
    This implements the meta-evaluation metric from the WMT Metrics Shared Task,
    as described in "Ties Matter: Meta-Evaluating Modern Metrics with Pairwise
    Accuracy and Tie Calibration" (Deutsch et al., 2023).
    
    The metric:
    1. Groups translations by their source sentence (exact match)
    2. For each group, calculates pairwise ranking accuracy between metric and gold
    3. Automatically calibrates a tie threshold (epsilon) to optimize accuracy
    4. Averages accuracy across all items (source sentences)
    
    Args:
        src_sentences: List of source sentences (used for grouping)
        gold_scores: List of gold quality scores (e.g., human MQM scores)
        metric_scores: List of metric scores to evaluate
        sample_rate: Proportion of pairs to sample for epsilon search (1.0 = all)
        return_details: If True, return additional details
        
    Returns:
        If return_details=False: (accuracy, optimal_epsilon)
        If return_details=True: (accuracy, optimal_epsilon, num_items, accuracies_per_item)
    """
    # Remove pairwise NaNs
    valid_indices = [
        i for i in range(len(gold_scores))
        if not np.isnan(gold_scores[i]) and not np.isnan(metric_scores[i])
    ]
    
    if not valid_indices:
        if return_details:
            return np.nan, 0.0, 0, {}
        return np.nan, 0.0
    
    # Group by source sentence
    gold_by_src = defaultdict(list)
    metric_by_src = defaultdict(list)
    
    for idx in valid_indices:
        src = src_sentences[idx]
        gold_by_src[src].append(gold_scores[idx])
        metric_by_src[src].append(metric_scores[idx])
    
    # Filter out groups with only one translation (can't compute pairwise accuracy)
    valid_items = [src for src in gold_by_src if len(gold_by_src[src]) >= 2]
    
    if not valid_items:
        if return_details:
            return np.nan, 0.0, 0, {}
        return np.nan, 0.0
    
    # Keep only valid items
    gold_by_src_filtered = {src: gold_by_src[src] for src in valid_items}
    metric_by_src_filtered = {src: metric_by_src[src] for src in valid_items}
    
    # Find optimal epsilon through tie calibration
    best_epsilon, _ = find_optimal_epsilon(
        gold_by_src_filtered, 
        metric_by_src_filtered, 
        sample_rate
    )
    
    # Calculate final accuracy with optimal epsilon
    accuracies_per_item = {}
    for src in valid_items:
        acc, num_pairs = pairwise_accuracy_with_ties(
            gold_by_src_filtered[src], 
            metric_by_src_filtered[src], 
            best_epsilon
        )
        if not np.isnan(acc):
            accuracies_per_item[src] = acc
    
    if not accuracies_per_item:
        if return_details:
            return np.nan, best_epsilon, 0, {}
        return np.nan, best_epsilon
    
    avg_accuracy = np.mean(list(accuracies_per_item.values()))
    num_items = len(accuracies_per_item)
    
    if return_details:
        return avg_accuracy, best_epsilon, num_items, accuracies_per_item
    
    return avg_accuracy, best_epsilon


def under_over_confidence_measure(gold_quality, qe_output):
    data_size = len(qe_output)
    
    if data_size == 0:
        return None, None
    
    gold_quality = min_max_scale(gold_quality)
    qe_output = min_max_scale(qe_output)

    if np.isnan(gold_quality).any() or np.isnan(qe_output).any():
        return np.nan, np.nan
    
    under_confidence = (qe_output < gold_quality).sum() / data_size
    over_confidence = (qe_output > gold_quality).sum() / data_size
    
    return under_confidence, over_confidence


def token_to_sentence_scores(token_level_scores, aggregate):
    """
    Read token-level scores from a file and aggregate them into sentence-level scores.
    :param token_level_scores: token-level scores
    :param aggregate: Aggregation method ('mean', 'sum', 'prod', 'median', 'min').
    :return: Tuple of token-level scores and aggregated sentence-level scores.
    """
    if aggregate == 'mean':
        sentence_level_scores = [np.mean(np.array(x)) if len(x) > 0 else 0 for x in token_level_scores]
    elif aggregate == 'sum':
        sentence_level_scores = [np.sum(np.array(x)) if len(x) > 0 else 0 for x in token_level_scores]
    elif aggregate == 'prod':
        sentence_level_scores = [np.prod(np.array(x)) if len(x) > 0 else 0 for x in token_level_scores]
    elif aggregate == 'median':
        sentence_level_scores = [np.median(np.array(x)) if len(x) > 0 else 0 for x in token_level_scores]
    elif aggregate == 'min':
        sentence_level_scores = [np.min(np.array(x)) if len(x) > 0 else 0 for x in token_level_scores]
    else:
        raise RuntimeError(f"Invalid value for aggregate: {aggregate}.")
    return sentence_level_scores

def remove_pairwise_nans(x, y):
    """
    Remove entries where either x or y is NaN.
    
    Parameters
    ----------
    x, y : array-like
        Input arrays of equal length.
        
    Returns
    -------
    x_clean, y_clean : np.ndarray
        Arrays with NaN-corresponding entries removed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def calculate_comet_ref_gold(src, pred_txt, ref, configs, output_dir, use_comet_cache):
    """Calculate COMET ref-based gold quality scores."""
    if ref is None:
        return None
    cache_path = f"{output_dir}/inference_{configs['dataname']}/{configs['comet_ref_based']}.txt"
    if os.path.isfile(cache_path) and use_comet_cache:
        print("Load pre-computed ref-based COMET ...")
        pseudo_gold_quality = load_text_file(cache_path)
        pseudo_gold_quality = [float(x) for x in pseudo_gold_quality]
    else:
        print("Running ref-based COMET  ...")
        comet_ref_based = load_comet_model(model_name=configs['comet_ref_based'])
        pseudo_gold_quality = comet_ref_based.predict(
            format_for_comet(src, pred_txt, ref), batch_size=4, gpus=1
        ).scores
        write_text_file(pseudo_gold_quality, cache_path)

    # Log the system-level quality, also with some traditional metrics
    wandb.log({
        f"{configs['dataname']}_{configs['comet_ref_based']}": np.mean(pseudo_gold_quality),
        f"{configs['dataname']}_BLEU": BLEU().corpus_score(pred_txt, [ref]).score,
        f"{configs['dataname']}_chrF2": CHRF().corpus_score(pred_txt, [ref]).score,
    })
    
    return pseudo_gold_quality


def calculate_comet_qe_baseline(src, pred_txt, configs, output_dir, use_comet_cache, 
                                 pseudo_gold_quality, human_gold_quality):
    """Calculate and eval supervised COMET QE baseline."""
    if configs["comet_qe_baseline"] == "None":
        return
    
    cache_path = f"{output_dir}/inference_{configs['dataname']}/{configs['comet_qe_baseline']}.txt"
    if os.path.isfile(cache_path) and use_comet_cache:
        print("Loading COMET QE baseline ...")
        qe_output = load_text_file(cache_path)
        qe_output = [float(x) for x in qe_output]
    else:
        print("Running COMET QE baseline ...")
        comet_qe_sys = load_comet_model(model_name=configs['comet_qe_baseline'])
        qe_output = comet_qe_sys.predict(
            format_for_comet(src, pred_txt), batch_size=4, gpus=1
        ).scores
        write_text_file(qe_output, cache_path)

    log_correlations(dataname=configs['dataname'], 
                    qe_output=qe_output, 
                    pseudo_gold_quality=pseudo_gold_quality, 
                    human_gold_quality=human_gold_quality,
                    qe_name=configs['comet_qe_baseline'],
                    src_sentences=None)


def calculate_llm_judge_gold(configs, output_dir):
    """Load LLM-as-a-Judge pseudo ground truth for non-MT datasets."""
    cache_path = f"{output_dir}/inference_{configs['dataname']}/qwen25_72B.txt"
    if os.path.isfile(cache_path):
        print(f"Load pre-computed LLM-as-a-judge score from {cache_path} ...")
        pseudo_gold_quality = load_text_file(cache_path)
        pseudo_gold_quality = [float(x) if x != "None" else np.nan for x in pseudo_gold_quality]

        # Log the system-level quality
        wandb.log({
            f"{configs['dataname']}_qwen25_72B": np.mean(pseudo_gold_quality)
        })
        return pseudo_gold_quality
    elif not configs.get('force_decoding'):
        print("Please compute LLM-as-a-Judge score as ground truth first!")
        exit(0)
    else:
        return None


def eval_self_judge_qe(configs, output_dir, pseudo_gold_quality, human_gold_quality):
    """Load and eval self-judge QE scores."""
    model_name = ''.join(c for c in configs['model_id'] if c.isalnum()).lower()
    cache_path = f"{output_dir}/inference_{configs['dataname']}/{model_name}.txt"
    if os.path.isfile(cache_path):
        print(f"Load pre-computed LLM Self Judge score from {cache_path} ...")
        self_judge_qe_score = load_text_file(cache_path)
        self_judge_qe_score = [float(x) if x != "None" else np.nan for x in self_judge_qe_score]
        log_correlations(dataname=configs['dataname'], 
                        qe_output=self_judge_qe_score, 
                        pseudo_gold_quality=pseudo_gold_quality, 
                        human_gold_quality=human_gold_quality,
                        qe_name="selfjudge", 
                        agg="",
                        src_sentences=None)


def eval_monte_carlo_qe(configs, output_dir, pseudo_gold_quality, human_gold_quality):
    """Load and eval monte-carlo QE scores."""
    all_seed_results_available = True
    all_seed_results = []
    for seed_i in range(1, 11):
        result_path = f"{output_dir}/inference_{configs['dataname']}/results_{seed_i}.json"
        if not os.path.isfile(result_path):
            all_seed_results_available = False
            break
        with open(result_path, 'r') as f:
            all_seed_results.append(json.load(f))

    if not all_seed_results_available:
        print("Not all seed results are available. Skipping Monte Carlo Sequence Entropy calculation.")
        return

    softmax_lprobs_seqs = [
        token_to_sentence_scores(token_level_scores=result_seed_i['log_scores'], aggregate="sum") 
        for result_seed_i in all_seed_results
    ]
    sigmoid_lprobs_seqs = [
        token_to_sentence_scores(token_level_scores=result_seed_i['confidence_log_scores'], aggregate="sum") 
        for result_seed_i in all_seed_results
    ]

    mc_qe_softmax = np.exp(
        np.asarray(softmax_lprobs_seqs).mean(axis=0)
    )
    mc_qe_sigmoid = np.exp(
        np.asarray(sigmoid_lprobs_seqs).mean(axis=0)
    )

    log_correlations(dataname=configs['dataname'], 
                    qe_output=mc_qe_softmax, 
                    pseudo_gold_quality=pseudo_gold_quality, 
                    human_gold_quality=human_gold_quality,
                    qe_name="mc_softmax", 
                    agg="",
                    src_sentences=None)
    
    log_correlations(dataname=configs['dataname'], 
                    qe_output=mc_qe_sigmoid, 
                    pseudo_gold_quality=pseudo_gold_quality, 
                    human_gold_quality=human_gold_quality,
                    qe_name="mc_sigmoid", 
                    agg="",
                    src_sentences=None)


def eval_model_scores_qe(results, configs, pseudo_gold_quality, human_gold_quality):
    """Eval scores from model inference results."""
    for k, v in results.items():
        if 'scores' in k:
            if 'log' in k:
                for aggregate in ["sum", "mean"]:
                    qe_output = token_to_sentence_scores(token_level_scores=v, aggregate=aggregate)
                    log_correlations(dataname=configs['dataname'], 
                                     qe_output=qe_output, 
                                     pseudo_gold_quality=pseudo_gold_quality, 
                                     human_gold_quality=human_gold_quality,
                                     qe_name=k, 
                                     agg=aggregate,
                                     src_sentences=None)
            else:
                for aggregate in ["prod", "mean"]:
                    qe_output = token_to_sentence_scores(token_level_scores=v, aggregate=aggregate)
                    log_correlations(dataname=configs['dataname'], 
                                     qe_output=qe_output, 
                                     pseudo_gold_quality=pseudo_gold_quality, 
                                     human_gold_quality=human_gold_quality,
                                     qe_name=k, 
                                     agg=aggregate,
                                     src_sentences=None)


def token_scores_to_char_labels(token_scores, mt_text, tokenizer, threshold, invert=False):
    """
    Convert token-level scores to character-level binary labels.
    
    Args:
        token_scores: List of scores for each token
        mt_text: The MT text string
        tokenizer: The tokenizer used
        threshold: Score threshold for classifying as error
        invert: If False (default), scores BELOW threshold are errors (lower = more likely error).
                If True, scores ABOVE threshold are errors (higher = more likely error, e.g., entropy).
        
    Returns:
        char_labels: List of 0/1 labels for each character (1 = error, 0 = correct)
    """
    # Get token offsets for the MT text
    encoding = tokenizer(mt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    
    # Initialize all characters as correct (0)
    char_labels = [0] * len(mt_text)
    
    # Convert token scores to binary labels and map to characters
    for token_idx, (start, end) in enumerate(offsets):
        if token_idx >= len(token_scores):
            break
        
        score = token_scores[token_idx]
        
        # Determine if this token is an error based on threshold
        if invert:
            is_error = score > threshold  # Higher score = error (e.g., entropy)
        else:
            is_error = score < threshold  # Lower score = error (e.g., confidence)
        
        if is_error:
            for char_idx in range(start, end):
                if char_idx < len(char_labels):
                    char_labels[char_idx] = 1
    
    return char_labels


def annotations_to_char_labels(annotations, mt_text_length):
    """
    Convert annotation spans to character-level binary labels.
    
    Args:
        annotations: List of annotation dicts with 'start' and 'end' keys
        mt_text_length: Length of the MT text
        
    Returns:
        char_labels: List of 0/1 labels for each character (1 = error, 0 = correct)
    """
    char_labels = [0] * mt_text_length
    
    for annotation in annotations:
        start = annotation['start']
        end = annotation['end']
        for char_idx in range(start, min(end, mt_text_length)):
            char_labels[char_idx] = 1
    
    return char_labels


def calculate_esa_f1(pred_char_labels_list, gold_char_labels_list):
    """
    Calculate precision, recall, and F1 score for error span detection at character level.
    
    This follows the WMT ESA evaluation methodology where:
    - True Positive (TP): Character predicted as error and is actually an error
    - False Positive (FP): Character predicted as error but is actually correct
    - False Negative (FN): Character not predicted as error but is actually an error
    
    Args:
        pred_char_labels_list: List of predicted char label lists (one per sentence)
        gold_char_labels_list: List of gold char label lists (one per sentence)
        
    Returns:
        Dict with precision, recall, f1, and TP/FP/FN counts
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_labels, gold_labels in zip(pred_char_labels_list, gold_char_labels_list):
        # Ensure same length
        min_len = min(len(pred_labels), len(gold_labels))
        pred_labels = pred_labels[:min_len]
        gold_labels = gold_labels[:min_len]
        
        for pred, gold in zip(pred_labels, gold_labels):
            if pred == 1 and gold == 1:
                total_tp += 1
            elif pred == 1 and gold == 0:
                total_fp += 1
            elif pred == 0 and gold == 1:
                total_fn += 1
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }


def char_labels_to_token_labels(char_labels, text, tokenizer):
    """
    Convert character-level labels to token-level labels.
    Uses the get_char_to_token_mapping concept: a token is marked as error if any character it contains is an error.
    
    Args:
        char_labels: List of 0/1 labels for each character (1 = error, 0 = correct)
        text: The text string
        tokenizer: The tokenizer used
        
    Returns:
        token_labels: List of 0/1 labels for each token (1 = error, 0 = correct)
    """
    # Get token offsets for the text
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    
    # Create token-level labels: a token is an error if any of its characters is an error
    token_labels = []
    for start, end in offsets:
        # Check if any character in this token span is marked as error
        is_token_error = any(char_labels[char_idx] == 1 for char_idx in range(start, end) if char_idx < len(char_labels))
        token_labels.append(1 if is_token_error else 0)
    
    return token_labels


def find_optimal_threshold_for_esa(token_scores_list, mt_texts, gold_char_labels_list, tokenizer, 
                                    thresholds=None, invert=False):
    """
    Find the optimal threshold that maximizes F1 score for error span detection.
    
    Args:
        token_scores_list: List of token score lists (one per sentence)
        mt_texts: List of MT text strings
        gold_char_labels_list: Pre-computed list of gold char label lists (one per sentence)
        tokenizer: The tokenizer used
        thresholds: List of thresholds to try. If None, use linspace from min to max scores.
        invert: If False (default), scores BELOW threshold are errors (lower = more likely error).
                If True, scores ABOVE threshold are errors (higher = more likely error, e.g., entropy).
        
    Returns:
        Tuple of (best_threshold, best_f1, all_results)
    """
    # Flatten all scores to determine threshold range
    all_scores = [s for scores in token_scores_list for s in scores if not np.isnan(s)]
    if not all_scores:
        return 0.5, 0.0, {}
    
    if thresholds is None:
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        thresholds = np.linspace(min_score, max_score, 20)
    
    best_f1 = -1
    best_threshold = thresholds[0]
    all_results = {}
    
    for threshold in thresholds:
        pred_char_labels_list = [
            token_scores_to_char_labels(scores, mt_text, tokenizer, threshold, invert)
            for scores, mt_text in zip(token_scores_list, mt_texts)
        ]
        
        results = calculate_esa_f1(pred_char_labels_list, gold_char_labels_list)
        all_results[threshold] = results
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_threshold = threshold
    
    return best_threshold, best_f1, all_results


def eval_model_scores_error_spans(results, dataname, mt_texts, annotations_list, tokenizer, output_dir, split=None, skip_omissions=False):
    """
    Evaluate model scores for error span prediction on MQM/ESA datasets.
    
    This function:
    1. Converts token-level model scores to character-level binary labels
    2. Calculates precision, recall, and F1 at both character and token levels
    3. Logs metrics to wandb
    4. Saves threshold search results to JSON for inspection
    5. Saves character-level and token-level gold and predicted labels to JSON
    
    Args:
        results: Dict containing inference results with token-level scores
        dataname: Name of the dataset
        mt_texts: List of MT text strings
        annotations_list: List of annotation lists (character-level error spans)
        tokenizer: The tokenizer used for the model
        output_dir: Output directory for saving results
        split: The dataset split, determines whether to find or retrieve threshold (only calculate optimal threshold on 'dev')
        skip_omissions: If True, skip sentences that contain at least one omission annotation (start == -1 or end == -1)
    """
    print(f"Evaluating error span prediction for {dataname}...")
    
    # Optionally skip sentences containing omissions
    if skip_omissions:
        def has_omission(annotations):
            return any(a['start'] == -1 or a['end'] == -1 for a in annotations)
        
        keep_indices = [i for i, annots in enumerate(annotations_list) if not has_omission(annots)]
        num_skipped = len(annotations_list) - len(keep_indices)
        print(f"  Skipping {num_skipped}/{len(annotations_list)} sentences containing omissions")
        
        mt_texts = [mt_texts[i] for i in keep_indices]
        annotations_list = [annotations_list[i] for i in keep_indices]
        # Also filter the results (token-level scores) to match
        results = {
            k: [v[i] for i in keep_indices] if isinstance(v, list) and len(v) == len(keep_indices) + num_skipped else v
            for k, v in results.items()
        }
    
    # Compute and log gold annotation statistics (independent of predictions)
    gold_char_labels_list = [
        annotations_to_char_labels(annotations, len(mt_text))
        for annotations, mt_text in zip(annotations_list, mt_texts)
    ]
    
    # Convert to token-level labels
    gold_token_labels_list = [
        char_labels_to_token_labels(char_labels, mt_text, tokenizer)
        for char_labels, mt_text in zip(gold_char_labels_list, mt_texts)
    ]
    
    # Character-level statistics
    total_chars = sum(len(labels) for labels in gold_char_labels_list)
    total_error_chars = sum(sum(labels) for labels in gold_char_labels_list)
    error_rate_char = total_error_chars / total_chars if total_chars > 0 else 0.0
    
    # Token-level statistics
    total_tokens = sum(len(labels) for labels in gold_token_labels_list)
    total_error_tokens = sum(sum(labels) for labels in gold_token_labels_list)
    error_rate_token = total_error_tokens / total_tokens if total_tokens > 0 else 0.0
    
    wandb.log({
        f"{dataname}/esa_total_chars": total_chars,
        f"{dataname}/esa_total_error_chars": total_error_chars,
        f"{dataname}/esa_error_rate_char": error_rate_char,
        f"{dataname}/esa_total_tokens": total_tokens,
        f"{dataname}/esa_total_error_tokens": total_error_tokens,
        f"{dataname}/esa_error_rate_token": error_rate_token,
    })
    print(f"  Gold statistics (character level): error_rate={error_rate_char:.4f} ({total_error_chars}/{total_chars} chars)")
    print(f"  Gold statistics (token level): error_rate={error_rate_token:.4f} ({total_error_tokens}/{total_tokens} tokens)")
    
    # Initialize output dicts for character-level and token-level labels
    esa_output_labels = {
        'gold_label_character_level': gold_char_labels_list,
    }
    esa_token_output_labels = {
        'gold_label_token_level': gold_token_labels_list,
    }
    
    # Evaluate each score type (exclude log scores)
    score_keys = [k for k in results.keys() if 'scores' in k and 'log' not in k]
    
    for score_key in score_keys:
        token_scores_list = results[score_key]
        
        if len(token_scores_list) != len(mt_texts):
            print(f"  Skipping {score_key}: length mismatch ({len(token_scores_list)} vs {len(mt_texts)})")
            continue
        
        # For entropy_scores, higher value = more uncertain = more likely error
        # For all other scores, lower value = more likely error
        invert = (score_key == 'entropy_scores')
        
        # Determine if we should find or retrieve the optimal threshold
        if split == 'dev':
            # Find optimal threshold and calculate character-level metrics
            best_threshold, best_f1_char, all_results = find_optimal_threshold_for_esa(
                token_scores_list, mt_texts, gold_char_labels_list, tokenizer, invert=invert
            )
            
            # Save threshold search results to JSON for later inspection
            all_results_serializable = {str(k): v for k, v in all_results.items()}
            esa_results_path = f"{output_dir}/inference_{dataname}/esa_threshold_search_{score_key}.json"
            with open(esa_results_path, 'w') as f:
                json.dump(all_results_serializable, f, indent=2)
            print(f"  Saved threshold search results to {esa_results_path}")
            
            # Log the optimal threshold to wandb
            wandb.log({f"{score_key}/esa_optimal_threshold": best_threshold})
        else:
            # Retrieve optimal threshold from wandb
            best_threshold = wandb.run.summary.get(f"{score_key}/esa_optimal_threshold")
            if best_threshold is None:
                print(f"  Warning: Could not retrieve optimal threshold for {score_key} from wandb. Set default threshold = 0.5")
                best_threshold = 0.5
            else:
                print(f"  Retrieved optimal threshold for {score_key}: {best_threshold}")
        
        # Get predicted character-level labels at optimal threshold
        pred_char_labels_list = [
            token_scores_to_char_labels(scores, mt_text, tokenizer, best_threshold, invert)
            for scores, mt_text in zip(token_scores_list, mt_texts)
        ]
        esa_output_labels[f'{score_key}_label_character_level'] = pred_char_labels_list
        
        # Convert to token-level labels
        pred_token_labels_list = [
            char_labels_to_token_labels(char_labels, mt_text, tokenizer)
            for char_labels, mt_text in zip(pred_char_labels_list, mt_texts)
        ]
        esa_token_output_labels[f'{score_key}_label_token_level'] = pred_token_labels_list
        
        # Get character-level results at optimal threshold
        char_results = calculate_esa_f1(pred_char_labels_list, gold_char_labels_list)
        
        # Calculate token-level metrics
        token_results = calculate_esa_f1(pred_token_labels_list, gold_token_labels_list)
        
        # Log both character-level and token-level metrics to wandb
        log_dict = {
            # Character-level
            f"{dataname}_{score_key}/esa_f1_char": char_results['f1'],
            f"{dataname}_{score_key}/esa_precision_char": char_results['precision'],
            f"{dataname}_{score_key}/esa_recall_char": char_results['recall'],
            f"{dataname}_{score_key}/esa_optimal_threshold": best_threshold,
            # Token-level
            f"{dataname}_{score_key}/esa_f1_token": token_results['f1'],
            f"{dataname}_{score_key}/esa_precision_token": token_results['precision'],
            f"{dataname}_{score_key}/esa_recall_token": token_results['recall'],
        }
        wandb.log(log_dict)
        
        print(f"  {score_key}:")
        print(f"    Character-level F1: {char_results['f1']:.4f} (threshold: {best_threshold:.4f})")
        print(f"    Character-level Precision: {char_results['precision']:.4f}, Recall: {char_results['recall']:.4f}")
        print(f"    Token-level F1: {token_results['f1']:.4f}")
        print(f"    Token-level Precision: {token_results['precision']:.4f}, Recall: {token_results['recall']:.4f}")
    
    # Save character-level labels to JSON
    esa_labels_path = f"{output_dir}/inference_{dataname}/esa_output_labels.json"
    with open(esa_labels_path, 'w') as f:
        json.dump(esa_output_labels, f)
    print(f"  Saved character-level labels to {esa_labels_path}")
    
    # Save token-level labels to JSON
    esa_token_labels_path = f"{output_dir}/inference_{dataname}/esa_token_output_labels.json"
    with open(esa_token_labels_path, 'w') as f:
        json.dump(esa_token_output_labels, f)
    print(f"  Saved token-level labels to {esa_token_labels_path}")


def log_correlations(dataname, qe_output, pseudo_gold_quality, human_gold_quality, qe_name, agg="", src_sentences=None):
    if len(qe_output) == 0:
        return
    
    if ('pawsx' in dataname) or ("gsm8k" in dataname) or ("truthfulqa" in dataname):
        binary_labels = True
        if "entropy" in qe_name:
            qe_output = np.array(qe_output)
            qe_output = (qe_output - qe_output.min()) / (qe_output.max() - qe_output.min())
    else:
        binary_labels = False
    
    if pseudo_gold_quality is not None:
        x, y = remove_pairwise_nans(qe_output, pseudo_gold_quality)
        under_confidence, over_confidence = under_over_confidence_measure(y, x)
        log_dict = {
            f"{dataname}_{qe_name}{agg}/sent_level_pearsonr": pearsonr(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_spearmanr": spearmanr(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_kendalltau": kendalltau(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_under_confidence": under_confidence,
            f"{dataname}_{qe_name}{agg}/sent_level_over_confidence": over_confidence
        }
        
        # Add group-by-item accuracy with tie calibration (WMT Metrics acc*eq)
        # Only compute if we have source sentences for grouping

        if src_sentences is not None and len(src_sentences) == len(qe_output):
            try:
                acc_eq, opt_epsilon = groupby_item_accuracy_with_tie_calibration(
                    src_sentences, 
                    np.array(pseudo_gold_quality, dtype=float), 
                    np.array(qe_output, dtype=float),
                    sample_rate=1.0
                )
                log_dict[f"{dataname}_{qe_name}{agg}/sent_level_acc_eq"] = acc_eq
                log_dict[f"{dataname}_{qe_name}{agg}/sent_level_acc_eq_epsilon"] = opt_epsilon
            except Exception as e:
                print(f"Warning: Could not compute acc_eq for pseudo_gold: {e}")
        
        if binary_labels and "log" not in qe_name:
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_bceloss"] = log_loss(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_roc_auc_score"] = roc_auc_score(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_average_precision_score"] = average_precision_score(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_brier_score_loss"] = brier_score_loss(y, x),
        wandb.log(log_dict)

    if human_gold_quality is not None:
        x, y = remove_pairwise_nans(qe_output, human_gold_quality)
        under_confidence, over_confidence = under_over_confidence_measure(y, x)
        log_dict = {
            f"{dataname}_{qe_name}{agg}/sent_level_pearsonr_humanGold": pearsonr(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_spearmanr_humanGold": spearmanr(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_kendalltau_humanGold": kendalltau(x, y)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_under_confidence_humanGold": under_confidence,
            f"{dataname}_{qe_name}{agg}/sent_level_over_confidence_humanGold": over_confidence
        }
        
        # Add group-by-item accuracy with tie calibration (WMT Metrics acc*eq)
        # Only compute if we have source sentences for grouping
        if src_sentences is not None and len(src_sentences) == len(qe_output):
            try:
                acc_eq, opt_epsilon = groupby_item_accuracy_with_tie_calibration(
                    src_sentences, 
                    np.array(human_gold_quality, dtype=float), 
                    np.array(qe_output, dtype=float),
                    sample_rate=1.0
                )
                log_dict[f"{dataname}_{qe_name}{agg}/sent_level_acc_eq_humanGold"] = acc_eq
                log_dict[f"{dataname}_{qe_name}{agg}/sent_level_acc_eq_epsilon_humanGold"] = opt_epsilon
            except Exception as e:
                print(f"Warning: Could not compute acc_eq for human_gold: {e}")
        
        if binary_labels and "log" not in qe_name:
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_bceloss_humanGold"] = log_loss(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_roc_auc_score_humanGold"] = roc_auc_score(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_average_precision_score_humanGold"] = average_precision_score(y, x),
            log_dict[f"{dataname}_{qe_name}{agg}/sent_level_brier_score_loss_humanGold"] = brier_score_loss(y, x),
        wandb.log(log_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-paths", type=str, nargs='+', required=True)
    parser.add_argument("--wandb-run-id", type=str, required=True)
    parser.add_argument("--use-comet-cache", action='store_true', 
                        help="Whether to use the cached COMET scores. Recalculate if not set.")

    args = parser.parse_args()
    print(args)

    set_seed(0)

    configs = load_yaml_files(args.config_file_paths)
    
    os.environ["WANDB_RUN_ID"] = args.wandb_run_id
    os.environ["WANDB_PROJECT"] = "confidence_head_llm"
    output_dir = f"output/{args.wandb_run_id}"
    print(f"Output dir set to {output_dir}")

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        id=args.wandb_run_id,
        config=configs,
        config_exclude_keys=['dataname'],  # to not overwrite the name of the training data
        dir=output_dir,
        resume="allow"
    )

    # Load data
    test_dataset = build_datasets(
        dataname=configs.get('dataname'),
        src_path=configs.get('test_src_path'),
        tgt_path=configs.get('test_tgt_path'),
        src_lang=configs.get('src_lang'),
        tgt_lang=configs.get('tgt_lang'),
        split=configs.get('split'),
        raw_text_string=True,
        mqm_deduplicate=configs.get('mqm_deduplicate'),
        mqm_filter_no_annotations=configs.get('mqm_filter_no_annotations', False)
    )

    src = list(test_dataset['src'])
    ref = list(test_dataset['ref']) if 'ref' in test_dataset.column_names else None
    mt = list(test_dataset['mt']) if 'mt' in test_dataset.column_names else None

    write_text_file(src, f"{output_dir}/inference_{configs['dataname']}/src.txt")
    if ref is not None:
        write_text_file(ref, f"{output_dir}/inference_{configs['dataname']}/ref.txt")
    if mt is not None:
        write_text_file(mt, f"{output_dir}/inference_{configs['dataname']}/mt.txt")

    # Load inference results
    with open(f"{output_dir}/inference_{configs['dataname']}/results.json", 'r') as f:
        results = json.load(f)

    human_gold_quality = None
    if configs.get('force_decoding') and os.path.isfile(f"{os.environ.get('ROOT_DIR')}/{configs.get('human_da_path')}"):
        # This is the setting where we do forced decoding on translations where human quality scores are available
        human_gold_quality = load_text_file(f"{os.environ.get('ROOT_DIR')}/{configs.get('human_da_path')}")
        human_gold_quality = [float(x) for x in human_gold_quality]
        write_text_file(human_gold_quality, f"{output_dir}/inference_{configs['dataname']}/human_gold.txt")
        wandb.log({
            f"{configs['dataname']}_humanScore": np.mean(human_gold_quality)
        })

    if ("wmt" in configs['dataname']) or ("ParaCrawl" in configs['dataname']) or ("biomqm" in configs['dataname']):
        # Specific to translation test sets (using COMET models as pseudo ground-truth and supervised baseline)
        pseudo_gold_quality = calculate_comet_ref_gold(
            src, results['pred_txt'], ref, configs, output_dir, args.use_comet_cache
        )
        calculate_comet_qe_baseline(
            src, results['pred_txt'], configs, output_dir, args.use_comet_cache,
            pseudo_gold_quality, human_gold_quality
        )
    else:
        # Not MT dataset, use LLM-as-a-Judge as pseudo ground truth
        pseudo_gold_quality = calculate_llm_judge_gold(configs, output_dir)

    # Load and eval self-judge QE
    eval_self_judge_qe(configs, output_dir, pseudo_gold_quality, human_gold_quality)
        
    # Load and eval monte-carlo QE
    eval_monte_carlo_qe(configs, output_dir, pseudo_gold_quality, human_gold_quality)

    # Eval scores from model inference
    eval_model_scores_qe(results, configs, pseudo_gold_quality, human_gold_quality)
    
    # Eval error span prediction for MQM/ESA datasets
    if 'annotations' in test_dataset.column_names:
        annotations_list = list(test_dataset['annotations'])
        tokenizer = AutoTokenizer.from_pretrained(configs['model_id'])
        eval_model_scores_error_spans(results, configs['dataname'], mt, annotations_list, tokenizer, output_dir, configs.get('split'), skip_omissions=True)
    

if __name__ == "__main__":
    main()