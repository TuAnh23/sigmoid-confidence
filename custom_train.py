from transformers import TrainingArguments, Trainer
import torch


class CustomTrainingArguments(TrainingArguments):
    def __init__(
        self, 
        negative_sampling_method="token_freq", 
        negative_sampling_avoid_dominant=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.negative_sampling_method = negative_sampling_method
        self.negative_sampling_avoid_dominant = negative_sampling_avoid_dominant


class CustomTrainer(Trainer):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # following Huggingface's docs when `num_items_in_batch` is not used in  compute_loss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        confidence_logits = outputs.get('confidence_logits')

        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels.view(-1) != self.model.tokenizer.pad_token_id  # ignore padding tokens
        shift_confidence_logits = confidence_logits[..., :-1, :].contiguous()


        # BCE loss for new head
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        # One-hot encode labels, necessary for BCE
        one_hot_targets = torch.nn.functional.one_hot(
            shift_labels.view(-1),         # flatten to [B*T]
            num_classes=shift_confidence_logits.size(-1)  # vocab size
        ).float()  # shape: [B*T, vocab_size]

        bce_loss = bce_loss_fn(
            shift_confidence_logits.view(-1, shift_confidence_logits.size(-1))[mask], 
            one_hot_targets[mask]
        )

        # TODO: include original loss at outputs.get('loss') if you dont freeze base model
        loss =  bce_loss

        return (loss, outputs) if return_outputs else loss
