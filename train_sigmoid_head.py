import argparse
from model_with_sigmoid_head import AutoModelForCausalLMWithSigmoidHead
from prepare_data import build_datasets
from transformers import set_seed, EarlyStoppingCallback
import os
import time
from custom_train import CustomTrainingArguments, CustomTrainer
import wandb
from utils import load_yaml_files

def main():
    parser = argparse.ArgumentParser(description="Train a sigmoid head for a model.")
    parser.add_argument("--config-file-paths", type=str, nargs='+', required=True)
    parser.add_argument("--wandb-run-id", type=str, default=None)

    args = parser.parse_args()
    print(args)

    if args.wandb_run_id is None:
        wandb_run_id = str(time.time_ns())
        print(
            "Warning: wandb_run_id is not passed in and will be generated. " \
            "If you are launching this script with multi GPU training, there will be 2 IDs generated for one run, which is not desired."
        )
    else:
        wandb_run_id = args.wandb_run_id
    output_dir = f"output/{wandb_run_id}"
    print(f"Output dir set to {output_dir}")

    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_PROJECT"] = "confidence_head_llm"  

    set_seed(0)

    configs = load_yaml_files(args.config_file_paths)

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        id=wandb_run_id,
        config=configs,
        dir=output_dir,
        resume="allow"
    )

    # Instead of loading base model to GPUs already with device_map="auto", load to CPU first to do the weight copies. The Trainer will handle moving it to GPUs afterwards
    model = AutoModelForCausalLMWithSigmoidHead(configs.get('model_id'), head_type=configs.get('head_type'))

    if configs.get('head_type') in ["new_unembedding_head", "new_unembedding_head_and_rescaling_head"]:
        # Preparation before training starts
        # Copy weights from original head (only on main process)
        if configs.get('init_sigmoid_head_from_softmax_head'):
            model.confidence_head.weight.data.copy_(
                model.base_model.lm_head.weight.data
            )

    # Freeze all parameters except the new head
    if configs.get('freeze_base_model'):
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.confidence_head.parameters():
            param.requires_grad = True


    train_dataset = build_datasets(
        dataname=configs.get('dataname'),
        tokenizer=model.tokenizer, 
        max_length=configs.get('max_length'),
        src_path=configs.get('train_src_path'), 
        tgt_path=configs.get('train_tgt_path'), 
        src_lang=configs.get('src_lang'), 
        tgt_lang=configs.get('tgt_lang'), 
        split='train'
    )

    eval_dataset = build_datasets(
        dataname=configs.get('dataname'),
        tokenizer=model.tokenizer, 
        max_length=configs.get('max_length'),
        src_path=configs.get('dev_src_path'),
        tgt_path=configs.get('dev_tgt_path'),
        src_lang=configs.get('src_lang'),
        tgt_lang=configs.get('tgt_lang'),
        split='dev'
    )

    training_args = CustomTrainingArguments(
        output_dir=output_dir,
        eval_steps=configs.get('eval_steps'),
        eval_strategy='steps',
        save_steps=configs.get('save_steps'),
        save_strategy='steps',
        logging_steps=10,
        logging_strategy='steps',
        logging_dir=f"{output_dir}/logs",
        learning_rate=1e-4, # 5e-5
        lr_scheduler_type="cosine",
        per_device_train_batch_size=configs.get('per_device_train_batch_size'),
        per_device_eval_batch_size=configs.get('per_device_eval_batch_size'),
        gradient_accumulation_steps=configs.get('gradient_accumulation_steps'),
        weight_decay=0.01,
        save_total_limit=3,
        bf16=True,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=wandb_run_id,
        remove_unused_columns=False,
        label_names=['labels'],
        negative_sampling=configs.get('negative_sampling'),
        negative_sampling_ratio=configs.get('negative_sampling_ratio'),
        negative_sampling_method=configs.get('negative_sampling_method'), 
        combine_neg_distribution=configs.get('combine_neg_distribution'), 
        negative_sampling_avoid_dominant=configs.get('negative_sampling_avoid_dominant'),
        temperature_neg_sampling_softmax=configs.get('temperature_neg_sampling_softmax'),
        weight_positive=configs.get('weight_positive'),
        freeze_base_model=configs.get('freeze_base_model'),
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if d.startswith("checkpoint-")]
    resume_from_checkpoint = False if not checkpoints else True

    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end_time = time.time()
    training_duration = end_time - start_time
    # Log training duration to wandb in format "DD-HH:MM:SS"
    training_duration = f"{int(training_duration // 86400):02d}-{time.strftime('%H:%M:%S', time.gmtime(training_duration))}"
    wandb.log({"training_duration": training_duration}) 


if __name__ == "__main__":
    main()
