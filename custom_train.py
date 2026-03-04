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
        mqm_training_mode=False,  # Enable MQM token-level training mode
        weight_for_negative_mqm=1.0,  # Weight multiplier for negative samples in MQM mode
        add_ranking_loss=False,  # Add ranking loss to push positive above dominant tokens
        ranking_loss_margin=0.0,  # Margin for the ranking loss
        ranking_loss_weight=1.0,  # Weight multiplier for the ranking loss
        find_dominant_kwargs={},
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
        self.find_dominant_kwargs = find_dominant_kwargs
        self.mqm_training_mode = mqm_training_mode
        self.weight_for_negative_mqm = weight_for_negative_mqm
        self.add_ranking_loss = add_ranking_loss
        self.ranking_loss_margin = ranking_loss_margin
        self.ranking_loss_weight = ranking_loss_weight


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

        if len(sparse_params) > 0:
            self.args.max_grad_norm = None  # TODO: hacky: gradient clipping does not work with sparse tensors
        
        if len(optimizers) == 1:
            self.optimizer = optimizers[0]
        elif len(optimizers) == 2:
            self.optimizer = DualOptimizer(optimizers[0], optimizers[1])
        else:
            raise RuntimeError()

        return self.optimizer

    def sampled_confidence_logits(self, hidden_states, original_logits, model, row_idx, col_idx):
        # hidden_states: [B*T, hidden_dim]
        # to_keep_mask: [B*T, vocab_size]

        if model.head_type == "new_unembedding_head":
            h_selected = hidden_states[row_idx]
            w_selected = model.confidence_head(col_idx)  # confidence_head.weight [vocab_size, hidden_dim]
            # w_selected = confidence_head.weight[col_idx]
            confidence_logits = torch.sum(h_selected * w_selected, dim=1)
        elif model.head_type == "rescaling_head":
            confidence_logits = model.confidence_head(original_logits[row_idx,col_idx].view(-1, 1)).view_as(original_logits[row_idx,col_idx])
        elif model.head_type == "new_unembedding_head_and_rescaling_head":
            h_selected = hidden_states[row_idx]
            w_selected = model.confidence_head(col_idx)  # confidence_head.weight [vocab_size, hidden_dim]
            # w_selected = confidence_head.weight[col_idx]
            confidence_logits = torch.sum(h_selected * w_selected, dim=1)

            confidence_logits = model.rescaling_head(confidence_logits.view(-1, 1)).view_as(confidence_logits)
        else:
            raise RuntimeError(f"Unknown head_type {model.head_type}")

        return confidence_logits

    def _forward_and_shift(self, model, inputs):
        """
        Forward pass through the model and shift tensors for next-token prediction.
        
        Returns:
            tuple: (shift_labels, shift_logits, shift_hidden_states, outputs)
        """
        with torch.no_grad():
            outputs = model(
                **inputs,
                compute_confidence_logits=False,
            )

        hidden_size = outputs["last_hidden_states"].size(-1)
        vocab_size = outputs["logits"].size(-1)
        shift_labels = inputs.get("labels")[..., 1:].reshape(-1)  # [B*T]
        shift_logits = outputs["logits"][..., :-1, :].reshape(-1, vocab_size)  # [B*T, vocab_size]
        shift_hidden_states = outputs["last_hidden_states"][..., :-1, :].reshape(-1, hidden_size)  # [B*T, hidden_size]

        return shift_labels, shift_logits, shift_hidden_states, outputs

    def _compute_loss_from_samples(self, model, outputs, shift_hidden_states, shift_logits, 
                                    row_idx, col_idx, targets, pos_weight=None, sample_weights=None):
        """
        Compute confidence logits for sampled positions and return BCE loss.
        Used when negative_sampling=True.
        
        Args:
            model: The model
            outputs: Model outputs dict (will be modified with final loss)
            shift_hidden_states: Hidden states [B*T, hidden_size]
            shift_logits: Original logits [B*T, vocab_size]
            row_idx: Row indices for sampled positions
            col_idx: Column indices for sampled positions
            targets: Target values (0 or 1) for each sampled position
            pos_weight: Optional positive class weight for BCE loss
            sample_weights: Optional per-sample weight tensor for BCE loss
            
        Returns:
            tuple: (loss, outputs)
        """
        # Calculate confidence logits for selected positions
        with torch.autocast("cuda", enabled=False):
            shift_confidence_logits = self.sampled_confidence_logits(
                shift_hidden_states,
                shift_logits,
                model,
                row_idx,
                col_idx
            )

        # BCE loss
        # shift_confidence_logits: [N], targets: [N], sample_weights: [N] or None, pos_weight: scalar or None
        # All three tensors share the same dimension N (one entry per sampled position)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(weight=sample_weights, pos_weight=pos_weight)
        bce_loss = bce_loss_fn(shift_confidence_logits, targets)

        # Combine with base model loss if not frozen
        if self.args.freeze_base_model:
            loss = bce_loss
        else:
            loss = bce_loss + outputs.get('loss')

        outputs['loss'] = loss
        return loss, outputs

    def _compute_loss_full_vocab(self, model, outputs, shift_hidden_states, shift_logits, shift_labels):
        """
        Compute confidence logits for full vocabulary and return BCE loss.
        Used when negative_sampling=False.
        
        Args:
            model: The model
            outputs: Model outputs dict (will be modified with final loss)
            shift_hidden_states: Hidden states [B*T, hidden_size]
            shift_logits: Original logits [B*T, vocab_size]
            shift_labels: Labels [B*T]
            
        Returns:
            tuple: (loss, outputs)
        """
        with torch.autocast("cuda", enabled=False):
            if model.head_type == "new_unembedding_head":
                shift_confidence_logits = torch.matmul(
                    shift_hidden_states,
                    model.confidence_head.weight.T
                )
            elif model.head_type == "rescaling_head":
                shift_confidence_logits = model.confidence_head(shift_logits.view(-1, 1)).view_as(shift_logits)
            elif model.head_type == "new_unembedding_head_and_rescaling_head":
                shift_confidence_logits = torch.matmul(
                    shift_hidden_states,
                    model.confidence_head.weight.T
                )
                shift_confidence_logits = model.rescaling_head(shift_confidence_logits.view(-1, 1)).view_as(shift_confidence_logits)
            else:
                raise RuntimeError(f"Unknown head_type {model.head_type}")

        # Ignore padding tokens
        non_padding_mask = shift_labels != self.model.tokenizer.pad_token_id
        shift_confidence_logits = shift_confidence_logits[non_padding_mask]
        one_hot_targets = torch.nn.functional.one_hot(
            shift_labels[non_padding_mask],
            num_classes=shift_logits.size(-1)
        ).float()

        # Calculate pos_weight
        if self.args.weight_positive == "balance":
            n_positive = (one_hot_targets == 1).sum()
            n_negative = (one_hot_targets == 0).sum()
            pos_weight = n_negative / n_positive
        else:
            pos_weight = None

        # BCE loss
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = bce_loss_fn(shift_confidence_logits, one_hot_targets)

        # Combine with base model loss if not frozen
        if self.args.freeze_base_model:
            loss = bce_loss
        else:
            loss = bce_loss + outputs.get('loss')

        outputs['loss'] = loss
        return loss, outputs


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Dispatch to MQM training mode if enabled
        if self.args.mqm_training_mode:
            return self.compute_loss_mqm(model, inputs, return_outputs, num_items_in_batch)
        
        # Forward pass and shift for next-token prediction
        shift_labels, shift_logits, shift_hidden_states, outputs = self._forward_and_shift(model, inputs)

        if self.args.negative_sampling:
            # Sample negative tokens
            return_dominant = self.args.add_ranking_loss and self.args.negative_sampling_avoid_dominant
            sampling_result = self.negative_sampling_mask(
                torch.nn.functional.log_softmax(shift_logits, dim=-1), 
                shift_labels,
                return_dominant=return_dominant
            )
            if return_dominant:
                neg_row_idx, neg_col_idx, dominant_row_indices, dominant_col_indices = sampling_result
            else:
                neg_row_idx, neg_col_idx = sampling_result

            # Ignore the padding tokens
            non_padding = shift_labels[neg_row_idx] != self.model.tokenizer.pad_token_id
            neg_row_idx = neg_row_idx[non_padding]
            neg_col_idx = neg_col_idx[non_padding]

            # Target is 1 if col_idx matches the label, 0 otherwise
            targets = (neg_col_idx == shift_labels[neg_row_idx]).float()
            pos_weight = torch.tensor(self.args.negative_sampling_ratio) if self.args.weight_positive == "balance" else None
            
            loss, outputs = self._compute_loss_from_samples(
                model, outputs, shift_hidden_states, shift_logits,
                neg_row_idx, neg_col_idx, targets, pos_weight
            )

            # Ranking loss: penalize when dominant (non-target) tokens are ranked above the true positive
            if return_dominant and dominant_row_indices is not None and len(dominant_row_indices) > 0:
                # Filter out dominant tokens that ARE the true label (we only penalize non-target dominant tokens)
                dom_labels_at_pos = shift_labels[dominant_row_indices]
                non_label_mask = dominant_col_indices != dom_labels_at_pos
                # Also filter out padding positions
                non_pad_mask = dom_labels_at_pos != self.model.tokenizer.pad_token_id
                keep_mask = non_label_mask & non_pad_mask
                dom_row = dominant_row_indices[keep_mask]
                dom_col = dominant_col_indices[keep_mask]

                if len(dom_row) > 0:
                    pos_col = shift_labels[dom_row]  # true label token for each position
                    with torch.autocast("cuda", enabled=False):
                        # Confidence logit for the true positive token
                        pos_confidence = self.sampled_confidence_logits(
                            shift_hidden_states, shift_logits, model, dom_row, pos_col
                        )
                        # Confidence logit for the dominant (non-target) token
                        dom_confidence = self.sampled_confidence_logits(
                            shift_hidden_states, shift_logits, model, dom_row, dom_col
                        )
                    # MarginRankingLoss: loss = max(0, -target * (x1 - x2) + margin)
                    # With target=1: loss = max(0, -(pos - dom) + margin) = max(0, dom - pos + margin)
                    ranking_target = torch.ones_like(pos_confidence)
                    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=self.args.ranking_loss_margin)
                    ranking_loss = ranking_loss_fn(pos_confidence, dom_confidence, ranking_target)
                    loss = loss + self.args.ranking_loss_weight * ranking_loss
                    outputs['loss'] = loss
        else:
            # Full vocabulary computation
            loss, outputs = self._compute_loss_full_vocab(
                model, outputs, shift_hidden_states, shift_logits, shift_labels
            )

        return (loss, outputs) if return_outputs else loss
    

    def compute_loss_mqm(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        MQM (Machine Translation Quality Metrics) Training Mode.
        
        Training behavior:
        - Tokens with mqm_label = 1 (correct): positive target + N sampled negatives
        - Tokens with mqm_label = 0 (erroneous): single negative only (no additional sampling)
        - Tokens with mqm_label = -1 (ignored): skip entirely
        
        Implementation notes:
        - We use lists (row_idx_list, col_idx_list, target_list) to collect samples from 
          two different sources: (1) correct tokens with negative sampling, and (2) erroneous 
          tokens as single negatives. These are concatenated at the end for a single loss computation.
        - mapped_row_idx is needed because negative_sampling_mask() returns indices relative to 
          the filtered correct_logits tensor (0 to len(correct_indices)-1), but we need indices 
          into the original shift_* tensors. We map back via: mapped_row_idx = correct_indices[neg_row_idx]
        """
        # Forward pass and shift for next-token prediction
        shift_labels, shift_logits, shift_hidden_states, outputs = self._forward_and_shift(model, inputs)
        
        # Get MQM token labels (shifted to align with predictions)
        shift_mqm_labels = inputs.get("mqm_token_labels")[..., 1:].reshape(-1)  # [B*T]
        
        # Create masks for different token types
        non_padding_mask = shift_labels != self.model.tokenizer.pad_token_id
        non_ignored_mask = shift_mqm_labels != -1
        valid_mask = non_padding_mask & non_ignored_mask
        
        correct_mask = valid_mask & (shift_mqm_labels == 1)
        error_mask = valid_mask & (shift_mqm_labels == 0)
        
        correct_indices = torch.where(correct_mask)[0]
        error_indices = torch.where(error_mask)[0]
        
        # Lists to accumulate samples from different sources before concatenation
        row_idx_list = []   # Indices into shift_* tensors (row dimension)
        col_idx_list = []   # Token IDs (column dimension / vocabulary)
        target_list = []    # Binary targets: 1 for correct, 0 for incorrect
        is_mqm_error_list = []  # Track which samples originate from MQM error spans (mqm_label=0)
        
        # Handle correct tokens (mqm_label = 1): positive + negative sampling
        if len(correct_indices) > 0:
            correct_labels = shift_labels[correct_indices]
            correct_logits = shift_logits[correct_indices]
            
            if self.args.negative_sampling:
                # negative_sampling_mask returns indices relative to correct_logits (0 to len-1)
                neg_row_idx, neg_col_idx = self.negative_sampling_mask(
                    torch.nn.functional.log_softmax(correct_logits, dim=-1),
                    correct_labels
                )
                # Map back to original shift_* tensor indices
                mapped_row_idx = correct_indices[neg_row_idx]
                row_idx_list.append(mapped_row_idx)
                col_idx_list.append(neg_col_idx)
                # Target is 1 only for the actual label token, 0 for sampled negatives
                target_list.append((neg_col_idx == correct_labels[neg_row_idx]).float())
                is_mqm_error_list.append(torch.zeros(len(mapped_row_idx), dtype=torch.bool, device=correct_labels.device))
            else:
                row_idx_list.append(correct_indices)
                col_idx_list.append(correct_labels)
                target_list.append(torch.ones(len(correct_indices), device=correct_labels.device))
                is_mqm_error_list.append(torch.zeros(len(correct_indices), dtype=torch.bool, device=correct_labels.device))
        
        # Handle erroneous tokens (mqm_label = 0): single negative only
        if len(error_indices) > 0:
            error_labels = shift_labels[error_indices]
            row_idx_list.append(error_indices)
            col_idx_list.append(error_labels)
            target_list.append(torch.zeros(len(error_indices), device=error_labels.device))
            is_mqm_error_list.append(torch.ones(len(error_indices), dtype=torch.bool, device=error_labels.device))
        
        # Edge case: no valid tokens
        if len(row_idx_list) == 0:
            loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            outputs['loss'] = loss
            return (loss, outputs) if return_outputs else loss
        
        row_idx = torch.cat(row_idx_list)          # [N] indices into shift_* tensors
        col_idx = torch.cat(col_idx_list)          # [N] token IDs (vocabulary indices)
        targets = torch.cat(target_list)           # [N] binary targets: 1=correct, 0=incorrect
        is_mqm_error = torch.cat(is_mqm_error_list)  # [N] bool: True only for MQM error-span tokens (mqm_label=0)
        
        # Calculate pos_weight for class imbalance
        if self.args.weight_positive == "balance":
            n_positive = (targets == 1).sum().float()
            n_negative = (targets == 0).sum().float()
            pos_weight = (n_negative / n_positive if n_positive > 0 else torch.tensor(1.0)).to(shift_logits.device)
        else:
            pos_weight = None
        
        # Per-sample weights: apply higher weight only to MQM error-span tokens (mqm_label=0)
        # sample_weights shape: [N], same as targets and shift_confidence_logits in _compute_loss_from_samples
        sample_weights = None
        if self.args.weight_for_negative_mqm != 1.0:
            sample_weights = torch.where(
                is_mqm_error,                       # [N] bool mask
                torch.tensor(self.args.weight_for_negative_mqm, device=targets.device),  # scalar broadcast to [N]
                torch.tensor(1.0, device=targets.device)                                  # scalar broadcast to [N]
            )  # [N] per-sample weights
        
        loss, outputs = self._compute_loss_from_samples(
            model, outputs, shift_hidden_states, shift_logits,
            row_idx, col_idx, targets, pos_weight, sample_weights
        )
        
        return (loss, outputs) if return_outputs else loss


    def create_sampling_distribution(self, single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices):
        if single_negative_sampling_method == "random":
            sampling_distribution = torch.ones_like(softmax_lprobs, dtype=torch.float)
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
        

    def negative_sampling_mask(self, softmax_lprobs, labels, return_dominant=False):
        """
        :param softmax_lprobs: lprobs output by the original softmax head
        :param labels: one-hot encoded LM labels
        :param return_dominant: if True, also return dominant row/col indices
        :return:
        """
        if self.args.negative_sampling_avoid_dominant:
            dominant_indices = find_dominant(softmax_lprobs, **self.args.find_dominant_kwargs)

            # Convert dominant_indices to a more convenient format
            # Create a mask for valid indices (i.e., where not -1)
            valid_mask = dominant_indices != -1

            # Get row indices (e.g., [0,0,1,1,1,...])
            row_indices = torch.arange(dominant_indices.shape[0], device=dominant_indices.device).unsqueeze(1).expand_as(dominant_indices)
            dominant_row_indices = row_indices[valid_mask]
            # Get column indices from dominant_indices, filtered by valid_mask
            dominant_col_indices = dominant_indices[valid_mask]
            del dominant_indices, valid_mask
        else:
            dominant_row_indices = None
            dominant_col_indices = None

        # Combine sampling distributions
        if self.args.combine_neg_distribution == "add":
            sampling_distribution = None
            for single_negative_sampling_method in self.args.negative_sampling_method.split(','):
                sampling_distribution = self.create_sampling_distribution(single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices) if sampling_distribution is None \
                                        else sampling_distribution + self.create_sampling_distribution(single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices)
            sampled_positions = self.efficient_sampling(sampling_distribution=sampling_distribution, num_samples=self.args.negative_sampling_ratio)
        elif self.args.combine_neg_distribution == "multiply":
            sampling_distribution = None
            for single_negative_sampling_method in self.args.negative_sampling_method.split(','):
                sampling_distribution = self.create_sampling_distribution(single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices) if sampling_distribution is None \
                                        else sampling_distribution * self.create_sampling_distribution(single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices)
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
                        sampling_distribution=self.create_sampling_distribution(single_negative_sampling_method, softmax_lprobs, dominant_row_indices, dominant_col_indices), 
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
        pos_row_idx = torch.arange(len(labels), device=labels.device)
        pos_col_idx = labels
        row_idx_list.append(pos_row_idx)
        col_idx_list.append(pos_col_idx)

        # Concatenate all indices
        row_idx = torch.cat(row_idx_list)
        col_idx = torch.cat(col_idx_list)

        # Deduplicate
        unique_pairs = torch.unique(torch.stack([row_idx, col_idx], dim=1), dim=0)
        row_idx = unique_pairs[:, 0]
        col_idx = unique_pairs[:, 1]
        
        if return_dominant:
            return row_idx, col_idx, dominant_row_indices, dominant_col_indices
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

