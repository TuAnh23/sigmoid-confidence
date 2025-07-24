import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AutoModelForCausalLMWithSigmoidHead(torch.nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name) # , torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
            print("pad_token added:", self.tokenizer.pad_token)
        else:
            print("pad_token already exists:", self.tokenizer.pad_token)

        # Create confidence head with the same shape as the original softmax head
        self.confidence_head = torch.nn.Linear(
            self.base_model.lm_head.in_features,
            self.base_model.lm_head.out_features,
            bias=False,
            device=self.base_model.device
        )

    def forward(self, *args, **kwargs):
        outputs = self.base_model(
            *args,
            **kwargs,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        with torch.autocast("cuda", enabled=False):
            # Train head with full precision
            confidence_logits = self.confidence_head(outputs.hidden_states[-1])  # confidence head logits

        return {
            "loss": outputs.get('loss'),
            "logits": outputs.get('logits'),
            "confidence_logits": confidence_logits
        }
