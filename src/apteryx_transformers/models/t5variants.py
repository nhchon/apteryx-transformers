from datetime import datetime
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils import data
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from ..collators import DataCollatorForAutoencodersBATCH


class T5EncoderAggDecoder(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
        block_size=512,
        encoding_vector_size=512,
        agg=True,
        agg_mode="linear",
    ):
        super(T5EncoderAggDecoder, self).__init__(config)
        self.block_size = block_size

        self.agg = agg
        self.agg_mode = agg_mode
        print(f'AGG: {self.agg}, MODE: {self.agg_mode if self.agg else "N/A"}')
        if self.agg and self.agg_mode == "linear":
            print("Using Linear Aggregation")
            self.encoding_vector_size = encoding_vector_size
            self.enc_to_vec = nn.Linear(
                self.block_size * self.config.d_model, encoding_vector_size
            )
            self.vec_to_enc_hat = nn.Linear(
                encoding_vector_size, self.block_size * self.config.d_model
            )

    def forward(
        self,
        agg=True,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if self.model_parallel:
            assert False, "This variant does not support model parallel."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)

        (
            encoder_outputs,
            dense_vector_encoding,
        ) = self.encode_w_agg(  # If encoder_outputs provided, will spit back out
            encoder_outputs=encoder_outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decode_custom(
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        return self.lm_head_step(
            sequence_output,
            decoder_outputs,
            encoder_outputs,
            labels=labels,
            return_dict=return_dict,
        )

    def lm_head_step(
        self,
        sequence_output,
        decoder_outputs,
        encoder_outputs,
        labels=None,
        return_dict=False,
    ):
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def encode_w_agg(
        self,
        encoder_outputs=None,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **unused_kwargs,
    ):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.agg:
                agg_outputs = self.temporal_agg(encoder_outputs, attention_mask)
                encoder_outputs = agg_outputs["output_formatted_for_decoder"]
                dense_vector_encoding = agg_outputs["dense_vector_encoding"]

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return encoder_outputs, dense_vector_encoding

    def temporal_agg(self, encoder_outputs, attention_masks):
        encoder_last_hidden_states = encoder_outputs.last_hidden_state
        batch_size, seq_len, d_model = encoder_last_hidden_states.shape

        result = {"output_formatted_for_decoder": None, "dense_vector_encoding": None}

        # Copy mask along d_model axis (the third one)
        masks_expanded = attention_masks[:, :, None].repeat(1, 1, d_model)

        # elementwise multiplication of enc last hidden state with mask,
        # which should remove irrelevant states (those which were masked) from the average.
        hidden_masked = encoder_last_hidden_states.mul(masks_expanded)

        if self.agg_mode == "mean":
            # Average, dividing by total unmasked tokens (time axis, e.g. the second)
            hidden_masked_summed = hidden_masked.sum(1)
            masked_time_agg = hidden_masked_summed.div(masks_expanded.sum(1))

            masked_time_agg_broadcast = masked_time_agg[:, None, :].repeat(
                1, seq_len, 1
            )

            # Make sure to put back the time axis!
            result["output_formatted_for_decoder"] = BaseModelOutput(
                last_hidden_state=masked_time_agg_broadcast,
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

            result["dense_vector_encoding"] = masked_time_agg

        elif self.agg_mode == "linear":
            # Stack each sequence embedding and project to an encoding vector
            stacked = hidden_masked.view(batch_size, seq_len * d_model)
            dense_vector_encoding = self.enc_to_vec(
                stacked
            )  # (batch_size, self.encoding_vector_size)
            result["dense_vector_encoding"] = dense_vector_encoding
            encoder_output_hat = self.vec_to_enc_hat(dense_vector_encoding).view(
                batch_size, seq_len, d_model
            )  # (batch_size, seq_len, d_model)

            result["output_formatted_for_decoder"] = BaseModelOutput(
                last_hidden_state=encoder_output_hat,
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return result

    def _shift_right(self, input_ids):
        # In T5 decoder_start_token_id is usually set to the pad_token_id.
        decoder_start_token_id = self.config.pad_token_id
        pad_token_id = self.config.pad_token_id

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def decode_custom(
        self,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):

        # Should always be the case?
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        return self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
