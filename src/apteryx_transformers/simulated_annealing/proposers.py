import numpy as np
import spacy
import torch
import logging
import string

from transformers import RobertaTokenizerFast, RobertaForMaskedLM


class WordLevelProposer:
    def __init__(
        self,
        tokenizer="roberta-base",
        model="roberta-base",
        spacy_model="en_core_web_sm",
        device="cpu",
        include_insert=True,
        include_delete=False,
        n_masks=1,
    ):
        self.device = device
        self.logger = logging.Logger("Proposer")
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer)
        self.model = RobertaForMaskedLM.from_pretrained(model).to(self.device)
        self.ops = [
            ("edit", self.edit),
            ("insert", self.insert) if include_insert else None,
            ("delete", self.delete) if include_delete else None,
        ]
        self.ops = [i for i in self.ops if i]
        self.n_masks = n_masks

    def propose(self, s):
        opname, op = self.ops[np.random.randint(len(self.ops))]
        return {"op": opname, "output": op(s)}

    def edit(self, s, allow_same=False):
        masked, inputs, mask_idx, original_word = self.mask(s, mode="edit")
        new = inputs.input_ids.clone()
        out = self.model(**inputs)
        probs = out.logits.softmax(2)

        # Set probs of original vocab tokens to zero
        if not allow_same:
            masked_ids = self.tokenizer(
                original_word, add_special_tokens=False
            ).input_ids
            probs[:, mask_idx, masked_ids] = 0

        edit = torch.multinomial(
            probs[:, mask_idx], num_samples=1, replacement=True
        ).item()
        new[:, mask_idx] = edit

        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[
            0
        ]  # , mask_idx, original_word

    def insert(self, s):
        masked, inputs, mask_idx, original_word = self.mask(s, mode="insert")
        new = inputs.input_ids.clone()
        out = self.model(**inputs)

        edit = torch.multinomial(
            out.logits.softmax(2)[:, mask_idx], num_samples=1, replacement=True
        ).item()
        new[:, mask_idx] = edit

        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[
            0
        ]  # , mask_idx, original_word

    def delete(self, s):
        """
        This MOSTLY handles dangling whitespace after deletion. Revisit at some point - likely not a huge deal though.
        """
        to_mask, start_idx, end_idx, pos, words = self.mask(s, mode="delete")

        left_char = s[start_idx - 1] if start_idx > 0 else ""
        right_char = s[end_idx] if end_idx < len(s) else ""

        n_whitespace_remaining = sum(
            [left_char in string.whitespace, right_char in string.whitespace]
        )

        if n_whitespace_remaining == 2:
            # Delete char to the right - it doesn't matter, both are whitespace
            return s[:start_idx] + s[end_idx + 1 :]  # , to_mask

        elif n_whitespace_remaining == 1:
            # Don't delete anything
            return s[:start_idx] + s[end_idx:]  # , to_mask

        elif n_whitespace_remaining == 0:
            # Replace missing whitespace
            return s[:start_idx] + " " + s[end_idx:]  # , to_mask
        else:
            print("Should never reach here.")
            return s[:start_idx] + s[end_idx:]  # , to_mask

        # elif left_char in string.whitespace:
        #     #Right char is either punct or part of another word; remove in both directions
        #     return s[:start_idx - 1] + s[end_idx:]
        # elif right_char in string.whitespace:
        #     #Left char is likely punct; leave both
        #     return s[:start_idx] + s[end_idx:]

    def mask(self, s, mode="insert"):
        # encoded = self.tokenizer(s, return_tensors='pt')
        # original = encoded.input_ids
        # input_ids = original.clone()
        #
        # n_tok = input_ids.shape[-1]
        # #Randomly set a token to MASK
        # mask_idx = np.random.randint(1, n_tok - 1)
        # input_ids[:, mask_idx] = self.tokenizer.mask_token_id
        # encoded.input_ids = input_ids
        # encoded.update({'labels': original})
        # return encoded, mask_idx

        """
        whole-word masking.
        """
        words = [
            (token.text, token.idx, token.idx + len(token.text), token.pos_)
            for token in self.nlp(s)
        ]
        word_idx = np.random.randint(len(words))
        to_mask, start_idx, end_idx, pos = words[word_idx]

        if mode == "delete":
            return to_mask, start_idx, end_idx, pos, words

        if mode == "insert":
            masked = s[:end_idx] + self.tokenizer.mask_token + s[end_idx:]

        elif mode == "edit":
            masked = s[:start_idx] + self.tokenizer.mask_token + s[end_idx:]

        else:
            self.logger.error(f'Mask mode "{mode}" not valid')
            assert False

        inputs = self.tokenizer(masked, return_tensors="pt").to(self.device)

        mask_idx = (
            (inputs.input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()
        )

        return masked, inputs, mask_idx, to_mask


class TokenLevelProposer:
    def __init__(
        self,
        tokenizer="roberta-base",
        model="roberta-base",
        device="cpu",
        include_insert=True,
        include_delete=False,
    ):
        self.device = device
        self.logger = logging.Logger("Proposer")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer)
        self.model = RobertaForMaskedLM.from_pretrained(model).to(self.device)
        self.ops = [
            "edit",
            "insert" if include_insert else None,
            "delete" if include_delete else None,
        ]
        self.ops = [i for i in self.ops if i]

    def propose(self, s, n_masks=5):
        opname = self.ops[np.random.randint(len(self.ops))]
        return {"op": opname, "output": self._propose(s, mode=opname, n_masks=n_masks)}

    def _propose(self, s, mode="edit", n_masks=5):
        inputs, chosen_idxs = self.mask(s, mode=mode, n_masks=n_masks)
        assert (
            inputs["input_ids"].shape[0] == 1
        ), "Did you accidentally pass in two strings?"

        if mode == "delete":
            return self.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )[0]
        else:
            out = self.model(**inputs)
            new = inputs["input_ids"].clone()

            probs = out.logits.softmax(2)
            mask_idxs = torch.nonzero(new == self.tokenizer.mask_token_id)

            # If there exist mask indexes:
            if mask_idxs.sum().item() != 0:
                mask_idxs = mask_idxs[:, 1]

                edits = torch.multinomial(
                    probs[0][mask_idxs], num_samples=1, replacement=True
                )

                new[0][mask_idxs] = edits.flatten()
                return self.tokenizer.batch_decode(new, skip_special_tokens=True)[0]
            else:
                # Nothing was masked, so simply return s.
                return s

    def insert_mask_at(self, t1, idx):
        return torch.cat(
            [t1[:, :idx], torch.Tensor([[self.tokenizer.mask_token_id]]), t1[:, idx:]],
            dim=1,
        ).long()

    def pop_at(self, t1, idxs_to_pop):
        all_idxs = np.arange(t1.shape[-1])
        return t1[:, [i.item() for i in all_idxs if i not in idxs_to_pop]]

    def mask(self, s, mode="insert", n_masks=5, max_sample_factor=0.5):
        encoded = self.tokenizer(s, return_tensors="pt")
        original = encoded.input_ids
        input_ids = original.clone()

        # Choose tokens from 1 to end - 1 (avoid padding)
        # Make sure you don't try to sample too many tokens!
        # Never sample more than 1/2 of all the tokens in the inputs. This could be a hyp.

        n_to_chose = min(n_masks, int(input_ids.shape[-1] * max_sample_factor))

        options = np.arange(1, input_ids.shape[1] - 1)
        chosen_idxs = (
            np.sort(np.random.choice(options, n_to_chose, replace=False))
            if len(options.ravel()) >= n_to_chose
            else np.array([0])
        )

        # Randomly set a token to MASK
        if mode == "edit":
            try:
                input_ids[:, chosen_idxs] = self.tokenizer.mask_token_id
            except:
                print("Failed EDIT op.")

        elif mode == "insert":
            try:
                # Work backwards (see [::-1] at end) so insertion doesn't cause offsets
                chosen_idxs = chosen_idxs[::-1]
                for idx in chosen_idxs:
                    input_ids = self.insert_mask_at(input_ids, idx)
            except:
                print("Failed INSERT op.")

        elif mode == "delete":
            try:
                input_ids = self.pop_at(input_ids, chosen_idxs)
            except:
                print("Failed DELETE op.")

        to_return = {
            "input_ids": input_ids.to(device=self.device),
            "attention_mask": torch.ones(input_ids.shape).to(device=self.device),
        }

        return to_return, chosen_idxs
