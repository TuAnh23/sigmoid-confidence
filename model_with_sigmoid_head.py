import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class AutoModelForCausalLMWithSigmoidHead(nn.Module):
    def __init__(self, base_model_name, device_map):
        super().__init__()
        # TODO only load in bfloat16 for quick debugging purpose
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=device_map, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Duplicate the lm_head
        self.confidence_head = nn.Linear(
            self.model.lm_head.in_features,
            self.model.lm_head.out_features,
            bias=False,
        )

        # Copy weights from original head
        self.confidence_head.weight.data.copy_(self.model.lm_head.weight.data)

        # Freeze all parameters except the new head TODO freeze depend on whether to train 2 losses or 1 
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.confidence_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        logits = self.model.lm_head(outputs.hidden_states[-1])  # original logits
        confidence_logits = self.confidence_head(outputs.hidden_states[-1])  # confidence head logits

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels.view(-1) != self.tokenizer.pad_token_id  # ignore padding tokens

            # Standard CE loss for original head
            # TODO: should we do it with label smoothing?
            ce_loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            ce_loss = ce_loss_fn(
                shift_logits.view(-1, shift_logits.size(-1))[mask], 
                shift_labels.view(-1)[mask]
            )

            # BCE loss for new head
            bce_loss_fn = nn.BCEWithLogitsLoss()
            shift_confidence_logits = confidence_logits[..., :-1, :].contiguous()
            # One-hot encode labels, necessary for BCE
            one_hot_targets = torch.nn.functional.one_hot(
                shift_labels.view(-1),         # flatten to [B*T]
                num_classes=shift_confidence_logits.size(-1)  # vocab size
            ).float()  # shape: [B*T, vocab_size]

            bce_loss = bce_loss_fn(
                shift_confidence_logits.view(-1, shift_confidence_logits.size(-1))[mask], 
                one_hot_targets[mask]
            )

            # TODO: include ce loss if you want to train 2 things along size 
            loss =  bce_loss

        return {
            "loss": loss,
            "logits": logits,
            "confidence_logits": confidence_logits
        }
