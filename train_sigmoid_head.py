import argparse
from model_with_sigmoid_head import AutoModelForCausalLMWithSigmoidHead
from prepare_data import build_datasets
from transformers import set_seed, EarlyStoppingCallback
import os
from custom_train import CustomTrainingArguments, CustomTrainer

def main():
    parser = argparse.ArgumentParser(description="Train a sigmoid head for a model.")
    parser.add_argument("--model-id", type=str, required=True, help="Huggingface model ID")
    parser.add_argument("--data", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory.")

    args = parser.parse_args()
    print(args)

    os.environ["WANDB_PROJECT"] = "confidence_head_llm"  

    set_seed(0)

    model = AutoModelForCausalLMWithSigmoidHead(args.model_id, device_map="auto")

    train_dataset, eval_dataset = build_datasets(model.tokenizer, args.model_id, max_length=2048)

    # Preparation before training starts
    # Copy weights from original head
    model.confidence_head.weight.data.copy_(model.base_model.lm_head.weight.data)

    # Freeze all parameters except the new head TODO freeze depend on whether to train 2 losses or 1 
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.confidence_head.parameters():
        param.requires_grad = True


    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        eval_steps=50,
        eval_strategy='steps',
        save_steps=50,
        save_strategy='steps',
        logging_steps=2,
        logging_strategy='steps',
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=5e-4,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        gradient_accumulation_steps=6,
        weight_decay=0.01,
        save_total_limit=3,
        # num_train_epochs=10,
        bf16=True, # TODO maybe train in full precision
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="test",
        remove_unused_columns=False,
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
