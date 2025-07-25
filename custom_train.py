from transformers import TrainingArguments, Trainer
import torch
from utils import find_dominant
from collections import Counter
from tqdm import tqdm


class CustomTrainingArguments(TrainingArguments):
    def __init__(
        self, 
        negative_sampling=True,
        negative_sampling_ratio=10,
        negative_sampling_method="random", 
        negative_sampling_avoid_dominant=True,
        weight_positive="balance",
        freeze_base_model=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.negative_sampling = negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        self.negative_sampling_method = negative_sampling_method
        self.negative_sampling_avoid_dominant = negative_sampling_avoid_dominant
        self.weight_positive = weight_positive
        self.freeze_base_model = freeze_base_model


class CustomTrainer(Trainer):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # following Huggingface's docs when `num_items_in_batch` is not used in  compute_loss()

        # Count token freq for negative sampling later
        def count_tokens_batch(batch):
            counter = Counter()
            for labels in batch["labels"]:
                counter.update(labels.tolist())
            # Convert to dict of lists so it's serializable
            return {"token_ids": [list(counter.keys())], "counts": [list(counter.values())]}
        token_counts_dataset = self.train_dataset.map(
            count_tokens_batch,
            batched=True,
            batch_size=1000,
            num_proc=50,
            remove_columns=self.train_dataset.column_names,
        )
        # Aggregate results
        final_counts = Counter()
        for row in tqdm(token_counts_dataset, desc="Aggregating counts"):
            final_counts.update(dict(zip(row["token_ids"].tolist(), row["counts"].tolist())))
        # Convert to tensor
        self.token_counter = torch.zeros(self.model.base_model.lm_head.out_features, dtype=torch.long)
        for token_id, count in final_counts.items():
            self.token_counter[token_id] = count
        # Zero out pad token
        self.token_counter[self.model.tokenizer.pad_token_id] = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the output logits
        outputs = model(**inputs)
        logits = outputs.get('logits')  # [B,T,vocab_size]
        confidence_logits = outputs.get('confidence_logits')

        # Get labels, take care of masking the pad tokens and shifting
        labels = inputs.get("labels")  # [B,T]
        shift_labels = labels[..., 1:].contiguous()
        to_keep_mask = shift_labels.view(-1) != self.model.tokenizer.pad_token_id  # ignore padding tokens
        shift_confidence_logits = confidence_logits[..., :-1, :].contiguous()
        shift_confidence_logits = shift_confidence_logits.view(-1, shift_confidence_logits.size(-1))  # Reshape to [B*T,vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # Reshape to [B*T,vocab_size]

        # One-hot encode labels, necessary for BCE
        one_hot_targets = torch.nn.functional.one_hot(
            shift_labels.view(-1),         # flatten to [B*T]
            num_classes=shift_confidence_logits.size(-1)  # vocab size
        ).float()  # shape: [B*T, vocab_size]


        # Sample negative tokens
        if self.args.negative_sampling:
            to_keep_mask = to_keep_mask.unsqueeze(1).expand(-1, logits.size(-1)) & self.negative_sampling_mask(
                torch.nn.functional.log_softmax(shift_logits, dim=-1), 
                one_hot_targets
            )
            
        # Calculate the pos_weight, i.e, n_negative / n_positive ratio in the whole batch
        if self.args.weight_positive == "balance":
            if self.args.negative_sampling:
                pos_weight = torch.tensor(self.args.negative_sampling_ratio)
            else:
                # confidence_target_ignore_pad = confidence_target.clone().int()
                # confidence_target_ignore_pad.masked_fill_(ignore_mask, -1)
                n_positive = (one_hot_targets[to_keep_mask] == 1).sum()
                n_negative = (one_hot_targets[to_keep_mask] == 0).sum()
                pos_weight = n_negative / n_positive
                print("Check if its correctly 1/vocab size")
                breakpoint()
        else:
            pos_weight = None

        # BCE loss for new head
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        

        bce_loss = bce_loss_fn(
            shift_confidence_logits.view(-1, shift_confidence_logits.size(-1))[to_keep_mask], 
            one_hot_targets[to_keep_mask]
        )

        if self.args.freeze_base_model:
            loss = bce_loss
        else:
            loss = bce_loss + outputs.get('loss')

        # Overwrite base_model loss with the new loss that consider the sigmoid head
        outputs['loss'] = loss

        return (loss, outputs) if return_outputs else loss
    

    def negative_sampling_mask(self, softmax_lprobs, labels):
        """
        :param softmax_lprobs: lprobs output by the original softmax head
        :param labels: one-hot encoded LM labels
        :return:
        """
        softmax_probs = torch.exp(softmax_lprobs)

        sampled_mask = torch.zeros_like(labels, dtype=torch.bool)
        if self.args.negative_sampling_method == "random":
            sampling_distribution = torch.ones_like(labels, dtype=torch.float)
        elif self.args.negative_sampling_method == "softmax_probs":
            sampling_distribution = softmax_probs
        elif self.args.negative_sampling_method == "token_freq":
            sampling_distribution = self.token_counter.to(softmax_probs).repeat(softmax_probs.shape[0], 1)
        elif self.args.negative_sampling_method == "token_freq,softmax_probs":
            sampling_distribution = softmax_probs + self.token_counter.to(softmax_probs).repeat(softmax_probs.shape[0], 1)
        else:
            raise RuntimeError(f"Unknown negative sampling method {self.args.negative_sampling_method}")
        

        if self.args.negative_sampling_avoid_dominant:
            dominant_indices = find_dominant(softmax_lprobs, find_dominant_method='difference_jump', p_jump=0.3, diff_cut=0.005)

            # Convert dominant_indices to a more convenient format
            # Create a mask for valid indices (i.e., where not -1)
            valid_mask = dominant_indices != -1
            # Get row indices (e.g., [0,0,1,1,1,...])
            row_indices = torch.arange(dominant_indices.shape[0], device=dominant_indices.device).unsqueeze(1).expand_as(dominant_indices)
            row_indices = row_indices[valid_mask]
            # Get column indices from dominant_indices, filtered by valid_mask
            col_indices = dominant_indices[valid_mask]

            # Don't sample the dominant tokens as negative samples, since we don't know whether they are viable translation options
            sampling_distribution[row_indices, col_indices] = 0

        sampled_positions = torch.multinomial(input=sampling_distribution, num_samples=self.args.negative_sampling_ratio, replacement=False)
        
        # # Sampling based on token freq more efficently
        # self.topk_freq_normalized = self.topk_freq_normalized.to(softmax_probs)
        # self.topk_freq_idx = self.topk_freq_idx.to(softmax_probs)
        # sampled_positions = torch.multinomial(
        #     input=self.topk_freq_normalized.expand(softmax_probs.size(0), -1),
        #     num_samples=self.args.negative_sampling_ratio, replacement=False
        # )
        # sampled_positions = self.topk_freq_idx[sampled_positions].long()

        sampled_mask.scatter_(dim=-1, index=sampled_positions, value=True)
        del sampled_positions

        # Put the positive samples back to the mask
        sampled_mask[labels == 1] = True

        return sampled_mask
