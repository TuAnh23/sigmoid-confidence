from transformers import TrainingArguments, Trainer, TrainerCallback

import torch
from boostedprob import find_dominant
from collections import Counter
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW, SparseAdam


class DualOptimizer(torch.optim.Optimizer):
    """
    A wrapper that combines two optimizers into one, so that HuggingFace Trainer
    or other frameworks see it as a single optimizer.
    """

    def __init__(self, optimizer1, optimizer2):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2

        # required by PyTorch Optimizer base class, though we won't use it
        defaults = {}
        param_groups = optimizer1.param_groups + optimizer2.param_groups
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss1 = self.optimizer1.step(closure)
        loss2 = self.optimizer2.step(closure)
        # return whichever loss makes sense — typically the first one
        return loss1 if loss1 is not None else loss2

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer1.zero_grad(set_to_none=set_to_none)
        self.optimizer2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict["optimizer1"])
        self.optimizer2.load_state_dict(state_dict["optimizer2"])



def compute_metrics(eval_pred, pad_token_id):
    # Huggingface trainer's `prediction_step` function turn the dict output by model forward() into a tuple
    # So in our case, `eval_pred.predictions` is a size-3 tuple, containing the logits, hidden_states, confidence_logits (verified)
    labels = eval_pred.label_ids
    confidence_scores = eval_pred.predictions[2]

    # Do proper shifting for next word prediction
    confidence_scores = confidence_scores[..., :-1, :]
    confidence_scores = confidence_scores.reshape(-1, confidence_scores.shape[-1])  # Reshape to [B*T,vocab_size]
    confidence_scores = -np.logaddexp(0, -confidence_scores)
    confidence_scores = confidence_scores.exp()
    labels = labels[..., 1:]
    labels = labels.reshape(-1)
    mask = labels != pad_token_id  
    labels = np.eye(confidence_scores.shape[-1])[labels]

    # Flatten for binary classification
    preds = (confidence_scores > 0.9).astype(int)
    labels = labels.astype(int)
    preds = preds[mask]
    labels = labels[mask]

    return {
        "nr_positive_preds_avg": preds.sum(axis=-1).mean(),
        "positive_label_detected": np.multiply(preds, labels).sum(axis=-1).mean(),
    }


class CustomTrainingArguments(TrainingArguments):
    def __init__(
        self, 
        negative_sampling=True,
        negative_sampling_ratio=10,
        negative_sampling_method="random", 
        combine_neg_distribution="add",
        negative_sampling_avoid_dominant=True,
        temperature_neg_sampling_softmax=1.0,
        weight_positive="balance",
        freeze_base_model=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.negative_sampling = negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        self.negative_sampling_method = negative_sampling_method
        self.combine_neg_distribution = combine_neg_distribution
        self.negative_sampling_avoid_dominant = negative_sampling_avoid_dominant
        self.temperature_neg_sampling_softmax = temperature_neg_sampling_softmax
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
        # Normalize to get a freq distribution 
        self.token_counter = self.token_counter / self.token_counter.sum()


    def create_optimizer(self):
        # First create our custom optimizer
        if self.optimizer is not None:
            return self.optimizer

        # Separate sparse and dense parameters
        sparse_params = []
        dense_params = []
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                # Detect if this module produces sparse gradients
                if isinstance(module, torch.nn.Embedding) and module.sparse:
                    sparse_params.append(param)
                else:
                    dense_params.append(param)

        # Create two optimizers
        optimizers = []
        if len(dense_params) > 0:
            optimizers.append(
                AdamW(
                    dense_params,
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            )

        if len(sparse_params) > 0:
            optimizers.append(
                SparseAdam(
                    sparse_params,
                    lr=self.args.learning_rate,
                )
            )

        if len(sparse_params) > 0 and len(dense_params) == 0:
            self.args.max_grad_norm = None  # TODO: hacky: gradient clipping does not work with sparse tensors
        
        if len(optimizers) == 1:
            self.optimizer = optimizers[0]
        elif len(optimizers) == 2:
            self.optimizer = DualOptimizer(optimizers[0], optimizers[1])
        else:
            raise RuntimeError()

        return self.optimizer

    def sampled_confidence_logits(self, hidden_states, confidence_head, row_idx, col_idx):
        # hidden_states: [B*T, hidden_dim]
        # to_keep_mask: [B*T, vocab_size]
        # confidence_head.weight [vocab_size, hidden_dim]

        h_selected = hidden_states[row_idx]
        w_selected = confidence_head(col_idx)
        # w_selected = confidence_head.weight[col_idx]

        logits = torch.sum(h_selected * w_selected, dim=1)

        return logits


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the original model umembedding output logits and last hidden state
        outputs = model(
            **inputs,
            compute_confidence_logits=False,
        )

        # Get labels, take care of masking the pad tokens and shifting
        labels = inputs.get("labels")  # [B,T]
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = outputs.get('logits')[..., :-1, :].contiguous()  # outputs.get('logits') has shape [B,T,vocab_size]
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # Reshape to [B*T,vocab_size]
        shift_hidden_states = outputs.get('last_hidden_states')[..., :-1, :].contiguous()  
        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))  # Reshape to [B*T,hidden_size]

        # One-hot encode labels, necessary for BCE
        one_hot_targets = torch.nn.functional.one_hot(
            shift_labels.view(-1),         # flatten to [B*T]
            num_classes=shift_logits.size(-1)  # vocab size
        ).float()  # shape: [B*T, vocab_size]

        # Sample negative tokens
        if self.args.negative_sampling:
            neg_row_idx, neg_col_idx =  self.negative_sampling_mask(
                torch.nn.functional.log_softmax(shift_logits, dim=-1), 
                one_hot_targets
            )

            # Ignore the padding tokens
            non_padding = shift_labels.view(-1)[neg_row_idx] != self.model.tokenizer.pad_token_id
            neg_row_idx = neg_row_idx[non_padding]
            neg_col_idx = neg_col_idx[non_padding]

            # Calculate the necessary confidence logits 
            with torch.autocast("cuda", enabled=False):
                # Full head forward with full precision
                shift_confidence_logits = self.sampled_confidence_logits(
                    shift_hidden_states, 
                    model.confidence_head, 
                    neg_row_idx,
                    neg_col_idx
                )
            # Filter the target 
            one_hot_targets = one_hot_targets[neg_row_idx,neg_col_idx]
        else:
            with torch.autocast("cuda", enabled=False):
                # Full head forward with full precision
                shift_confidence_logits = model.confidence_head(shift_hidden_states)
            # Ignore padding tokens
            non_padding_mask = shift_labels.view(-1) != self.model.tokenizer.pad_token_id  # [B*T], bool
            one_hot_targets = one_hot_targets[non_padding_mask]  # [N_non_pad, vocab_size]
            shift_confidence_logits = shift_confidence_logits[non_padding_mask]
            
        # Calculate the pos_weight, i.e, n_negative / n_positive ratio in the whole batch
        if self.args.weight_positive == "balance":
            if self.args.negative_sampling:
                pos_weight = torch.tensor(self.args.negative_sampling_ratio)
            else:
                n_positive = (one_hot_targets == 1).sum()
                n_negative = (one_hot_targets == 0).sum()
                pos_weight = n_negative / n_positive
                print("Check if its correctly 1/vocab size")
                breakpoint()
        else:
            pos_weight = None

        # BCE loss for new head
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        bce_loss = bce_loss_fn(
            shift_confidence_logits, 
            one_hot_targets,
        )

        if self.args.freeze_base_model:
            loss = bce_loss
        else:
            # TODO: check the scale of the two losses
            breakpoint()
            loss = bce_loss + outputs.get('loss')

        # Overwrite base_model loss with the new loss that consider the sigmoid head
        outputs['loss'] = loss

        return (loss, outputs) if return_outputs else loss
    

    def create_sampling_distribution(self, single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices):
        if single_negative_sampling_method == "random":
            sampling_distribution = torch.ones_like(labels, dtype=torch.float)
        elif single_negative_sampling_method == "softmax_probs":
            sampling_distribution = torch.nn.functional.softmax(softmax_lprobs / self.args.temperature_neg_sampling_softmax, dim=-1)
        elif single_negative_sampling_method == "token_freq":
            sampling_distribution = self.token_counter.to(softmax_lprobs).repeat(softmax_lprobs.shape[0], 1)
        else:
            raise RuntimeError(f"Unknown negative sampling method {single_negative_sampling_method}")
        
        if dominant_row_indices is not None and dominant_col_indices is not None:
            # Don't sample the dominant tokens as negative samples, since we don't know whether they are viable translation options
            sampling_distribution[dominant_row_indices, dominant_col_indices] = 0

        return sampling_distribution
        

    def negative_sampling_mask(self, softmax_lprobs, labels):
        """
        :param softmax_lprobs: lprobs output by the original softmax head
        :param labels: one-hot encoded LM labels
        :return:
        """
        sampled_mask = torch.zeros_like(labels, dtype=torch.bool)

        if self.args.negative_sampling_avoid_dominant:
            dominant_indices = find_dominant(softmax_lprobs, find_dominant_method='difference_jump', p_jump=0.3, epsilon=0.005)

            # Convert dominant_indices to a more convenient format
            # Create a mask for valid indices (i.e., where not -1)
            valid_mask = dominant_indices != -1
            # Get row indices (e.g., [0,0,1,1,1,...])
            row_indices = torch.arange(dominant_indices.shape[0], device=dominant_indices.device).unsqueeze(1).expand_as(dominant_indices)
            dominant_row_indices = row_indices[valid_mask]
            # Get column indices from dominant_indices, filtered by valid_mask
            dominant_col_indices = dominant_indices[valid_mask]
        else:
            dominant_row_indices = None
            dominant_col_indices = None

        # Combine sampling distributions
        if self.args.combine_neg_distribution == "add":
            sampling_distribution = None
            for single_negative_sampling_method in self.args.negative_sampling_method.split(','):
                sampling_distribution = self.create_sampling_distribution(single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices) if sampling_distribution is None \
                                        else sampling_distribution + self.create_sampling_distribution(single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices)
            sampled_positions = self.efficient_sampling(sampling_distribution=sampling_distribution, num_samples=self.args.negative_sampling_ratio)
        elif self.args.combine_neg_distribution == "multiply":
            sampling_distribution = None
            for single_negative_sampling_method in self.args.negative_sampling_method.split(','):
                sampling_distribution = self.create_sampling_distribution(single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices) if sampling_distribution is None \
                                        else sampling_distribution * self.create_sampling_distribution(single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices)
            sampled_positions = self.efficient_sampling(sampling_distribution=sampling_distribution, num_samples=self.args.negative_sampling_ratio)
        elif self.args.combine_neg_distribution == "independent":
            nr_distributions = len(self.args.negative_sampling_method.split(','))
            sampled_positions = []
            for i, single_negative_sampling_method in enumerate(self.args.negative_sampling_method.split(',')):
                if i == nr_distributions - 1:
                    # Take the rest
                    nr_samples = self.args.negative_sampling_ratio - i * (self.args.negative_sampling_ratio//nr_distributions)
                else:
                    nr_samples = self.args.negative_sampling_ratio//nr_distributions

                sampled_positions.append(
                    self.efficient_sampling(
                        sampling_distribution=self.create_sampling_distribution(single_negative_sampling_method, labels, softmax_lprobs, dominant_row_indices, dominant_col_indices), 
                        num_samples=nr_samples)
                )
            sampled_positions = torch.cat(sampled_positions, dim=-1)
        else:
            raise NotImplementedError()
        
        # Gather the indices of the kept positions
        row_idx_list = []
        col_idx_list = []

        # Extract indices of sampled positions
        row_indices = torch.arange(sampled_positions.shape[0], device=labels.device).unsqueeze(1).repeat(1, sampled_positions.shape[1])
        row_idx_list.append(row_indices.reshape(-1))
        col_idx_list.append(sampled_positions.reshape(-1))
        del sampled_positions

        # Add positive label positions
        pos_row_idx, pos_col_idx = torch.nonzero(labels == 1, as_tuple=True)
        row_idx_list.append(pos_row_idx)
        col_idx_list.append(pos_col_idx)

        # Concatenate all indices
        row_idx = torch.cat(row_idx_list)
        col_idx = torch.cat(col_idx_list)

        # Deduplicate
        pairs = torch.stack([row_idx, col_idx], dim=1)  
        unique_pairs = torch.unique(pairs, dim=0)
        row_idx = unique_pairs[:, 0]
        col_idx = unique_pairs[:, 1]

        return row_idx, col_idx


    def efficient_sampling(self, sampling_distribution, num_samples, cut_off_k=None):
        if cut_off_k is None or cut_off_k > sampling_distribution.shape[-1]:
            return torch.multinomial(
                input=sampling_distribution, 
                num_samples=num_samples, 
                replacement=False
            )
        else:
            # topk is done per batch element
            topk_probs, topk_indices = torch.topk(sampling_distribution, k=cut_off_k, dim=-1)
            # sample within top-k
            sampled_local = torch.multinomial(topk_probs, num_samples=num_samples, replacement=False)
            # map back to global vocab indices
            sampled_positions = torch.gather(topk_indices, dim=-1, index=sampled_local)
            return sampled_positions

