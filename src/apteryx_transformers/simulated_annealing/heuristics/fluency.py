import numpy as np
import torch
from torch.nn.functional import softmax


from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
import logging


class FluencyScorer:
    def __init__(self, tok_dir=None, model_dir=None, base_type="gpt2", device="cpu"):
        logger = logging.Logger("Fluency Scorer")
        if tok_dir and not model_dir:
            logger.warning(
                f"Custom tokenizer specified ({tok_dir}), but a generic model is being used. Please reconsider unless you know what you're doing."
            )
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            tok_dir if tok_dir else base_type
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            model_dir if model_dir else base_type
        ).to(device)

    def __call__(self, sentences, stride=10):
        if not isinstance(sentences, list):
            sentences = [sentences]
        return [
            {"perplexity": self.perplexity(s, stride=stride), "sequence": s}
            for s in sentences
        ]

    def perplexity(self, s, stride=10):
        # Thanks huggingface! https://huggingface.co/transformers/perplexity.html
        encodings = self.tokenizer(s, return_tensors="pt")
        max_length = self.model.config.n_positions

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            # Use a sliding window to calculate next-token log_likelihood.
            # Smaller windows are more accurate
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # outputs[0] is loss in this case
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)
        if lls:
            ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            return ppl.item()
        else:
            # Perplexity cannot be calculated on the empty string.
            # Set to infinity to strongly disincentivise.
            return np.inf
