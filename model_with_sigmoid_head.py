import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AutoModelForCausalLMWithSigmoidHead(torch.nn.Module):
    def __init__(self, base_model_name, device_map):
        super().__init__()
        # TODO only load in bfloat16 for quick debugging purpose
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=device_map, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Create confidence head with the same shape as the original softmax head
        self.confidence_head = torch.nn.Linear(
            self.base_model.lm_head.in_features,
            self.base_model.lm_head.out_features,
            bias=False,
        )

    def forward(self, *args, **kwargs):
        outputs = self.base_model(
            *args,
            **kwargs,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        confidence_logits = self.confidence_head(outputs.hidden_states[-1])  # confidence head logits

        return {
            "loss": outputs.get('loss'),
            "logits": outputs.get('logits'),
            "confidence_logits": confidence_logits
        }
