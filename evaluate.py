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


def min_max_scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) if np.min(arr) != np.max(arr) else np.nan

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

def log_correlations(dataname, qe_output, pseudo_gold_quality, human_gold_quality, qe_name, agg=""):
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
        raw_text_string=True
    )

    src = list(test_dataset['src'])
    ref = list(test_dataset['ref'])

    # Load inference results
    with open(f"{output_dir}/inference_{configs['dataname']}/results.json", 'r') as f:
        results = json.load(f)


    human_gold_quality = None
    if configs.get('force_decoding'):
        # This is the setting where we do forced decoding on translations where human quality scores are available
        human_gold_quality = load_text_file(f"{os.environ.get('ROOT_DIR')}/{configs.get('human_da_path')}")
        human_gold_quality = [float(x) for x in human_gold_quality]
        wandb.log({
            f"{configs['dataname']}_humanScore": np.mean(human_gold_quality)
        })

    if ("wmt" in configs['dataname']) or ("ParaCrawl" in configs['dataname']) or ("biomqm" in configs['dataname']):
        # Specific to translation test sets (using COMET models as pseudo ground-truth and supervised baseline)
        # Calculate COMET ref-based gold quality
        cache_path = f"{output_dir}/inference_{configs['dataname']}/{configs['comet_ref_based']}.txt"
        if os.path.isfile(cache_path) and args.use_comet_cache:
            print("Load pre-computed ref-based COMET ...")
            pseudo_gold_quality = load_text_file(cache_path)
            pseudo_gold_quality = [float(x) for x in pseudo_gold_quality]
        else:
            print("Running ref-based COMET  ...")
            comet_ref_based = load_comet_model(model_name=configs['comet_ref_based'])
            pseudo_gold_quality = comet_ref_based.predict(
                format_for_comet(src, results['pred_txt'], ref), batch_size=4, gpus=1
            ).scores
            write_text_file(pseudo_gold_quality, cache_path)

        # Log the system-level quality, also with some traditional metrics
        wandb.log({
            f"{configs['dataname']}_{configs['comet_ref_based']}": np.mean(pseudo_gold_quality),
            f"{configs['dataname']}_BLEU": BLEU().corpus_score(results['pred_txt'], [ref]).score,
            f"{configs['dataname']}_chrF2": CHRF().corpus_score(results['pred_txt'], [ref]).score,
        })

        # Calculate and eval supervised baseline
        if configs["comet_qe_baseline"] != "None":
            cache_path = f"{output_dir}/inference_{configs['dataname']}/{configs['comet_qe_baseline']}.txt"
            if os.path.isfile(cache_path) and args.use_comet_cache:
                print("Loading COMET QE baseline ...")
                qe_output = load_text_file(cache_path)
                qe_output = [float(x) for x in qe_output]
            else:
                print("Running COMET QE baseline ...")
                comet_qe_sys = load_comet_model(model_name=configs['comet_qe_baseline'])
                qe_output = comet_qe_sys.predict(
                    format_for_comet(src, results['pred_txt']), batch_size=4, gpus=1
                ).scores
                write_text_file(qe_output, cache_path)

        log_correlations(dataname=configs['dataname'], 
                        qe_output=qe_output, 
                        pseudo_gold_quality=pseudo_gold_quality, 
                        human_gold_quality=human_gold_quality,
                        qe_name=configs['comet_qe_baseline'])
    else:
        # Not MT dataset, use LLM-as-a-Judge as pseudo ground truth
        # Load ref-based gold quality
        cache_path = f"{output_dir}/inference_{configs['dataname']}/qwen25_72B.txt"
        if os.path.isfile(cache_path):
            print(f"Load pre-computed LLM-as-a-judge score from {cache_path} ...")
            pseudo_gold_quality = load_text_file(cache_path)
            pseudo_gold_quality = [float(x) if x != "None" else np.nan for x in pseudo_gold_quality]

            # Log the system-level quality
            wandb.log({
                f"{configs['dataname']}_qwen25_72B": np.mean(pseudo_gold_quality)
            })
        elif not configs.get('force_decoding'):
            print("Please compute LLM-as-a-Judge score as ground truth first!")
            exit(0)
        else:
            pseudo_gold_quality = None


    # Load and eval self-judge QE
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
                        agg="")
        
    # Load and eval monte-carlo QE
    all_seed_results_available = True
    all_seed_results = []
    for seed_i in range(1, 11):
        result_path = f"{output_dir}/inference_{configs['dataname']}/results_{seed_i}.json"
        if not os.path.isfile(result_path):
            all_seed_results_available = False
            break
        # Load inference results
        with open(result_path, 'r') as f:
            all_seed_results.append(json.load(f))

    if not all_seed_results_available:
        print("Not all seed results are available. Skipping Monte Carlo Sequence Entropy calculation.")
    else:
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
                        agg="")
        
        log_correlations(dataname=configs['dataname'], 
                        qe_output=mc_qe_sigmoid, 
                        pseudo_gold_quality=pseudo_gold_quality, 
                        human_gold_quality=human_gold_quality,
                        qe_name="mc_sigmoid", 
                        agg="")

    # Eval scores from model inference
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
                                     agg=aggregate)
            else:
                for aggregate in ["prod", "mean"]:
                    qe_output = token_to_sentence_scores(token_level_scores=v, aggregate=aggregate)
                    log_correlations(dataname=configs['dataname'], 
                                     qe_output=qe_output, 
                                     pseudo_gold_quality=pseudo_gold_quality, 
                                     human_gold_quality=human_gold_quality,
                                     qe_name=k, 
                                     agg=aggregate)
    

if __name__ == "__main__":
    main()