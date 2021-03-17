import numpy as np
import spacy
import torch
import logging
import string

from transformers import RobertaTokenizerFast, RobertaForMaskedLM


class Proposer:
    def __init__(self, tokenizer='roberta-base', model='roberta-base', spacy_model='en_core_web_sm'):
        self.logger = logging.Logger('Proposer')
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer)
        self.model = RobertaForMaskedLM.from_pretrained(model)

    def edit(self, s, allow_same=False):
        masked, inputs, mask_idx, original_word = self.mask(s, mode='edit')
        new = inputs.input_ids.clone()
        out = self.model(**inputs)
        probs = out.logits.softmax(2)

        # Set probs of original vocab tokens to zero
        if not allow_same:
            masked_ids = self.tokenizer(original_word, add_special_tokens=False).input_ids
            probs[:, mask_idx, masked_ids] = 0

        edit = torch.multinomial(probs[:, mask_idx], num_samples=1, replacement=True).item()
        new[:, mask_idx] = edit

        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[0], mask_idx

    def insert(self, s):
        masked, inputs, mask_idx, _ = self.mask(s, mode='insert')
        new = inputs.input_ids.clone()
        out = self.model(**inputs)

        edit = torch.multinomial(out.logits.softmax(2)[:, mask_idx], num_samples=1, replacement=True).item()
        new[:, mask_idx] = edit

        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[0], mask_idx

    def delete(self, s):
        '''
        This MOSTLY handles dangling whitespace after deletion. Revisit at some point - likely not a huge deal though.
        '''
        to_mask, start_idx, end_idx, pos, words = self.mask(s, mode='delete')

        print(to_mask)
        left_char = s[start_idx - 1] if start_idx > 0 else ''
        right_char = s[end_idx] if end_idx < len(s) else ''
        print(left_char, '|', right_char)

        if (left_char in string.whitespace) and (right_char in string.whitespace):
            # Delete char to the right - it doesn't matter, both are whitespace
            return s[:start_idx] + s[end_idx + 1:]

        elif left_char in string.whitespace:
            #Right char is either punct or part of another word; remove in both directions
            return s[:start_idx - 1] + s[end_idx:]
        elif right_char in string.whitespace:
            #Left char is likely punct; leave both
            return s[:start_idx] + s[end_idx:]

        return s[:start_idx] + s[end_idx:]

    def mask(self, s, mode='insert'):
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

        '''
        whole-word masking.
        '''
        words = [(token.text, token.idx, token.idx + len(token.text), token.pos_) for token in self.nlp(s)]
        word_idx = np.random.randint(len(words))
        to_mask, start_idx, end_idx, pos = words[word_idx]

        if mode == 'delete':
            return to_mask, start_idx, end_idx, pos, words

        if mode == 'insert':
            masked = s[:end_idx] + self.tokenizer.mask_token + s[end_idx:]

        elif mode == 'edit':
            masked = s[:start_idx] + self.tokenizer.mask_token + s[end_idx:]

        else:
            self.logger.error(f'Mask mode "{mode}" not valid')
            assert False

        inputs = self.tokenizer(masked, return_tensors="pt")

        mask_idx = (inputs.input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()

        return masked, inputs, mask_idx, to_mask
