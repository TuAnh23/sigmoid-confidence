import torch
from safetensors.torch import load_file
from model_with_sigmoid_head import AutoModelForCausalLMWithSigmoidHead
import wandb
from transformers import set_seed
import argparse
import os
from utils import load_yaml_files, get_best_checkpoint, find_eos_idx, check_is_dominant
from prepare_data import build_datasets
from torch.utils.data import DataLoader
import time
from collections import defaultdict
from tqdm import tqdm 
import json


def main():
    parser = argparse.ArgumentParser(description="Train a sigmoid head for a model.")
    parser.add_argument("--config-file-paths", type=str, nargs='+', required=True)
    parser.add_argument("--wandb-run-id", type=str, required=True)
    parser.add_argument("--manual-inspect", action='store_true', help="If set, will enter a pdb session in every generation step for manual inspection.")

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
        dir=output_dir,
        resume="allow"
    )

    # 1. Load model and tokenizer
    model = AutoModelForCausalLMWithSigmoidHead(configs['model_id'])

    checkpoint_path = get_best_checkpoint(output_dir)
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Prepare test data
    test_dataset = build_datasets(
        dataname=configs.get('dataname'),
        tokenizer=model.tokenizer, 
        max_length=configs.get('max_length')//2, # since this is only the input prompt
        src_path=configs.get('test_src_path'),
        tgt_path=configs.get('test_tgt_path'),
        src_lang=configs.get('src_lang'),
        tgt_lang=configs.get('tgt_lang'),
        teacher_forcing=False
    )

    test_dataloader = DataLoader(test_dataset, batch_size=configs['per_device_test_batch_size'], shuffle=False)

    # 3. Run inference
    # Initialize a dict to store results, where the default values are empty lists
    results = defaultdict(list)

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Inference Progress"):
            # Ensure batch is moved to correct device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate outputs
            # Note: `output_hidden_states=True` is necessary to get the hidden states for the confidence head
            # `return_dict_in_generate=True` allows us to access the hidden states for later use on the heads
            # outputs.hidden_states[generation_step][decoder_layer] shape is [batch_size, 1, hidden_size],
            # except for the first token where the hidden states of all input tokens is also stored, so [batch_size, input_length, hidden_size]
            outputs = model.base_model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True,
                return_dict_in_generate=True, 
                output_scores=True,
                do_sample=True,
                num_beams=1,
                temperature=1.0,
                top_p=0.95,
                max_length=configs['max_length']
            )
            predicted_ids = outputs['sequences']  # [batch_size, total_max_length]
            predicted_ids = predicted_ids[:, batch['input_ids'].shape[-1]:] # [batch_size, generation_max_length]

            # Prepare last hidden states
            last_hidden_states = [
                x[-1]  # last layer
                for x in outputs.hidden_states
            ]
            # We need special care for the first generated token
            last_hidden_states[0] = last_hidden_states[0][:, -1:, :]
            # Each entry in last_hidden_states now have shape [batch_size, 1, hidden_size]
            # We stack them to [batch_size, generation_max_length, hidden_size]
            last_hidden_states = torch.cat(last_hidden_states, dim=1)  

            # Pass last hidden states to the two heads
            confidence_logits = model.confidence_head(last_hidden_states) 
            confidence_log_scores = torch.nn.functional.logsigmoid(confidence_logits)

            logits = model.base_model.lm_head(last_hidden_states) 
            log_scores = torch.nn.functional.log_softmax(logits, dim=-1)

            # Calculate the entropy of the log softmax
            entropy = torch.mul(log_scores, torch.exp(log_scores)).sum(dim=-1)

            # Calculate boosted prob of the log softmax
            boosted_prob = check_is_dominant(log_scores, predicted_ids)
            log_boosted_prob = torch.log(boosted_prob + 1e-10)  # Add small value to avoid log(0)

            # Gather the scores of the predicted tokens
            pred_log_scores = torch.gather(log_scores, -1, predicted_ids.unsqueeze(-1)).squeeze(-1)
            pred_confidence_log_scores = torch.gather(confidence_log_scores, -1, predicted_ids.unsqueeze(-1)).squeeze(-1)

            # Log scores to scores
            pred_scores = torch.exp(pred_log_scores).squeeze(-1)
            pred_confidence_scores = torch.exp(pred_confidence_log_scores).squeeze(-1)

            # Store output
            results['pred_txt'].extend(model.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True))
            for batch_item in range(predicted_ids.shape[0]):
                end_idx = find_eos_idx(predicted_ids[batch_item], model.tokenizer.eos_token_id)
                results['pred_tokenized_txt'].append(
                    model.tokenizer.convert_ids_to_tokens(predicted_ids[batch_item][:end_idx])
                )
                results['confidence_scores'].append(pred_confidence_scores[batch_item][:end_idx].cpu().tolist())
                results['confidence_log_scores'].append(pred_confidence_log_scores[batch_item][:end_idx].cpu().tolist())
                results['scores'].append(pred_scores[batch_item][:end_idx].cpu().tolist())
                results['log_scores'].append(pred_log_scores[batch_item][:end_idx].cpu().tolist())
                results['entropy_scores'].append(entropy[batch_item][:end_idx].cpu().tolist())
                results['boosted_prob_scores'].append(boosted_prob[batch_item][:end_idx].cpu().tolist())
                results['log_boosted_prob_scores'].append(log_boosted_prob[batch_item][:end_idx].cpu().tolist())


                if args.manual_inspect:
                    for j in range(end_idx):
                        print(f"SOURCE: {model.tokenizer.decode(batch['input_ids'][batch_item], skip_special_tokens=True)}")
                        print(f"PRED: {model.tokenizer.decode(predicted_ids[batch_item], skip_special_tokens=True)}")
                        print(f"PREFIX PRED: {model.tokenizer.decode(predicted_ids[batch_item][:j], skip_special_tokens=True)}")
                        print(f"TOKEN: {results['pred_tokenized_txt'][-1][j]}")
                        print(f"CONFIDENCE SCORE: {results['confidence_scores'][-1][j]}")
                        print(f"PROB SCORE: {results['scores'][-1][j]}")
                        print(f"Nr conf > 0.8: {(confidence_log_scores.exp()[batch_item][j] > 0.8).sum()}")
                        print(f"Top k tokens by main head: {model.tokenizer.convert_ids_to_tokens(log_scores.exp()[batch_item][j].topk(k=10).indices)}")
                        print(f"Top k tokens by conf head: {model.tokenizer.convert_ids_to_tokens(confidence_log_scores.exp()[batch_item][j].topk(k=10).indices)}")
                        print(f"Conf scores of main head top k: {confidence_log_scores.exp()[batch_item][j][log_scores.exp()[batch_item][j].topk(k=10).indices]}")
                        breakpoint()
                

    end_time = time.time()
    inference_duration = end_time - start_time
    # Log training duration to wandb in format "DD-HH:MM:SS"
    inference_duration = f"{int(inference_duration // 86400):02d}-{time.strftime('%H:%M:%S', time.gmtime(inference_duration))}"
    wandb.log({f"{configs['dataname']}/inference_duration": inference_duration}) 

    # Save results to file
    os.makedirs(f"{output_dir}/inference_{configs['dataname']}", exist_ok=True)
    with open(f"{output_dir}/inference_{configs['dataname']}/results.json", 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()



