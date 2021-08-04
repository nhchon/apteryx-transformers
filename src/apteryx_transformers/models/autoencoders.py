from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union
import os

import torch
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
from .t5variants import T5EncoderAggDecoder


class AbstractTransformerAutoencoder(ABC):
    def __init__(
        self,
        dataset,
        model_name: str,
        model_config_dict: dict,
        training_args_dict: dict,
        block_size: int,
        tokenizer=None,
        encoding_vector_size=512,
        agg=True,
        agg_mode="linear",
        train_pct: float = 0.8,
        n_layers_to_train: Union[tuple, int] = 0,
    ):
        """

        :param dataset: The PyTorch Dataset used for training.
        :param model_name: The huggingface model name (e.g. 't5-small')
        :param model_config_dict: A dict with model config args; will be passed to the config constructor.
        :param training_args_dict: A dict with training arguments; if nothing is provided, will use default training arguments.
        :param block_size: The input size of the model (in tokens).
        :param train_pct: The percentage of the dataset to use for training; the remainder will be used for eval.
        :param n_layers_to_train: The number of attention layers to train. If a tuple, specifies for encoder and decoder separately.
        """
        self.model_name = model_name
        self.model_name_is_path = os.path.exists(self.model_name)
        if self.model_name_is_path:
            print(f"Loading model checkpoint from {self.model_name}")

        self.tokenizer = (
            tokenizer
            if tokenizer
            else self.get_tokenizer_class().from_pretrained(self.model_name)
        )
        self.collator = self.get_collator_class()()

        self.dataset = dataset
        self.block_size = self.dataset.block_size
        self.n_layers_to_train = n_layers_to_train

        if self.model_name_is_path:
            self.model = self.get_model_class().from_pretrained(
                self.model_name,
                block_size=self.block_size,
                encoding_vector_size=encoding_vector_size,
                agg=agg,
                agg_mode=agg_mode,
            )
            self.config = self.model.config
        else:
            self.config = (
                self.get_config_class()(**model_config_dict)
                if model_config_dict
                else self.get_config_class()()
            )
            self.config.decoder_start_token_id = self.tokenizer.pad_token_id

            # print(type(self.config))
            print(self.config)

            self.model = self.get_model_class()(
                config=self.config,
                block_size=self.block_size,
                encoding_vector_size=encoding_vector_size,
                agg=agg,
                agg_mode=agg_mode,
            )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.n_enc_layers = len(self.model.encoder.block)
        self.n_dec_layers = len(self.model.decoder.block)
        self.total_attn_layers = self.n_enc_layers + self.n_dec_layers

        if isinstance(self.n_layers_to_train, tuple):
            if self.n_layers_to_train == (-1, -1):
                # In the case of (-1, -1), EVERYTHING requires grad.
                for p in self.model.parameters():
                    p.requires_grad = True
            else:
                n_enc_to_train, n_dec_to_train = self.n_layers_to_train
                assert n_enc_to_train <= self.n_enc_layers
                assert n_dec_to_train <= self.n_dec_layers

                self.toggle_layer_grad(
                    self.model.encoder.block,
                    n_enc_to_train,
                    layer_acc=self.n_dec_layers,
                )
                self.toggle_layer_grad(self.model.decoder.block, n_dec_to_train)

        elif isinstance(self.n_layers_to_train, int) and self.n_layers_to_train > 0:
            assert (
                self.total_attn_layers >= self.n_layers_to_train
            ), f"You must select a number of layers to train less than the total number of layers in the model. You selected {self.n_layers_to_train}; there are {self.total_attn_layers} attention layers available."
            n_dec_to_train = (
                self.n_layers_to_train
                if self.n_layers_to_train < self.n_dec_layers
                else self.n_dec_layers
            )
            n_enc_to_train = max(self.n_layers_to_train - self.n_dec_layers, 0)

            self.toggle_layer_grad(
                self.model.encoder.block, n_enc_to_train, layer_acc=self.n_dec_layers
            )
            self.toggle_layer_grad(self.model.decoder.block, n_dec_to_train)

        else:
            # Turn all grads off except the final classification layer
            for i, p in enumerate(self.model.parameters()):
                # print(f'Layer {i}: OFF')
                p.requires_grad = False

        print(
            f"Autoencoder initialized; training lm_head and {self.n_layers_to_train} Attention Layers."
        )

        self.training_args = training_args_dict

        self.train_pct = train_pct

    def toggle_layer_grad(self, module, n_to_train, layer_acc=0):
        n_layers = len(module)

        # Train all layers if n_to_train is -1.
        if n_to_train == -1:
            n_to_train = n_layers

        if n_to_train > 0:
            for layer in module[:-n_to_train]:
                print(f"Layer {layer_acc}: OFF")
                for p in layer.parameters():
                    p.requires_grad = False
                layer_acc += 1
            for layer in module[-n_to_train:]:
                print(f"Layer {layer_acc}: ON")
                for p in layer.parameters():
                    p.requires_grad = True
                layer_acc += 1

    def get_trainer(self):

        now = datetime.now()

        # Can use dicts and instantiate with ** instead of below:
        default_training_args = TrainingArguments(
            output_dir=f"{self.model_name.split('/')[-1]}-{now.day}-{now.month}-{now.year}",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            save_steps=100,
            save_total_limit=2,
            logging_steps=20,
            dataloader_num_workers=15,
            warmup_steps=50,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=100,
            fp16=True,  # enable low-precision via AMP
            gradient_accumulation_steps=5,
        )
        if self.training_args:
            # Update default training args with specified model training args.
            for k, v in self.training_args.items():
                print(f'Updating Trainer["{k}"] with: {v}.')
                setattr(default_training_args, k, v)

        self.training_args = default_training_args

        N = len(self.dataset)
        n_train = int(N * self.train_pct)
        n_val = N - n_train
        print(f"Training on {n_train} examples;")
        print(f"Validating on {n_val} examples.")
        train_ds, eval_ds = data.random_split(self.dataset, (n_train, n_val))

        trainer = Trainer(
            model=self.model,
            args=default_training_args,
            data_collator=self.collator,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        return trainer

    def train(self, checkpoint_dir=None):
        trainer = self.get_trainer()
        if checkpoint_dir:
            trainer.train(checkpoint_dir)
        else:
            trainer.train()

    @abstractmethod
    def get_config_class(self):
        pass

    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def get_tokenizer_class(self):
        pass

    @abstractmethod
    def get_collator_class(self):
        pass


class T5AutoEncoder(AbstractTransformerAutoencoder):
    def __init__(
        self,
        dataset,
        model_name="t5-base",
        block_size=1024,
        encoding_vector_size=512,
        tokenizer=None,
        agg=True,
        agg_mode="linear",
        model_config_dict=None,
        training_args_dict=None,
        train_pct=0.8,
        n_layers_to_train=(-1, -1),
    ):
        super().__init__(
            dataset=dataset,
            model_name=model_name,
            model_config_dict=model_config_dict,
            training_args_dict=training_args_dict,
            block_size=block_size,
            encoding_vector_size=encoding_vector_size,
            tokenizer=tokenizer,
            agg=agg,
            agg_mode=agg_mode,
            train_pct=train_pct,
            n_layers_to_train=n_layers_to_train,
        )

    def get_model_class(self):
        return T5EncoderAggDecoder

    def get_tokenizer_class(self):
        return T5TokenizerFast

    def get_config_class(self):
        return T5Config

    def get_collator_class(self):
        return DataCollatorForAutoencodersBATCH
