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


def min_max_scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def under_over_confidence_measure(gold_quality, qe_output):
    data_size = len(qe_output)
    
    if data_size == 0:
        return None, None
    
    gold_quality = min_max_scale(gold_quality)
    qe_output = min_max_scale(qe_output)
    
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

def log_correlations(dataname, qe_output, gold_quality, qe_name, agg=""):
    if len(qe_output) == 0:
        return
    under_confidence, over_confidence = under_over_confidence_measure(gold_quality, qe_output)
    wandb.log(
        {
            f"{dataname}_{qe_name}{agg}/sent_level_pearsonr": pearsonr(qe_output, gold_quality)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_spearmanr": spearmanr(qe_output, gold_quality)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_kendalltau": kendalltau(qe_output, gold_quality)[0],
            f"{dataname}_{qe_name}{agg}/sent_level_under_confidence": under_confidence,
            f"{dataname}_{qe_name}{agg}/sent_level_over_confidence": over_confidence
        }
    )

def main():
    parser = argparse.ArgumentParser(description="Train a sigmoid head for a model.")
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

    # Calculate ref-based gold quality
    cache_path = f"{output_dir}/inference_{configs['dataname']}/{configs['comet_ref_based']}.txt"
    if os.path.isfile(cache_path) and args.use_comet_cache:
        print("Load pre-computed ref-based COMET ...")
        gold_quality = load_text_file(cache_path)
        gold_quality = [float(x) for x in gold_quality]
    else:
        print("Running ref-based COMET  ...")
        comet_ref_based = load_comet_model(model_name=configs['comet_ref_based'])
        gold_quality = comet_ref_based.predict(
            format_for_comet(src, results['pred_txt'], ref), batch_size=4, gpus=1
        ).scores
        write_text_file(gold_quality, cache_path)

    # Log the system-level quality, also with some traditional metrics
    wandb.log({
        f"{configs['dataname']}_{configs['comet_ref_based']}": np.mean(gold_quality),
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

    log_correlations(configs['dataname'], qe_output, gold_quality, configs['comet_qe_baseline'])

    # Eval scores from model inference
    for k, v in results.items():
        if 'scores' in k:
            if 'log' in k:
                for aggregate in ["sum", "mean"]:
                    qe_output = token_to_sentence_scores(token_level_scores=v, aggregate=aggregate)
                    log_correlations(configs['dataname'], qe_output, gold_quality, k, aggregate)
            else:
                for aggregate in ["prod", "mean"]:
                    qe_output = token_to_sentence_scores(token_level_scores=v, aggregate=aggregate)
                    log_correlations(configs['dataname'], qe_output, gold_quality, k, aggregate)
    

if __name__ == "__main__":
    main()