from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils import data

from apteryx_transformers.collators import DataCollatorForDocumentClassificationBATCH
from apteryx_transformers.apteryx_datasets import PickleDatasetFromDisk
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import dill as pickle
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix


class AbstractMultiClassifier(ABC):
    def __init__(
        self,
        data_dir,
        model_name,
        block_size,
        training_args,
        ds_limit,
        early_stopping_params,
        train_pct=0.8,
        data_dir_is_ds=False,
        n_layers_to_train=0,
    ):
        self.model_name = model_name
        self.block_size = block_size
        self.n_layers_to_train = n_layers_to_train
        self.tokenizer = self.get_tokenizer_class().from_pretrained(self.model_name)
        self.collator = DataCollatorForDocumentClassificationBATCH()
        if not data_dir_is_ds:
            self.dataset = PickleDatasetFromDisk(
                data_dir=data_dir,
                tokenizer=self.tokenizer,
                block_size=self.block_size,
                limit=ds_limit,
            )
        else:
            self.dataset = data_dir

        self.config = LongformerConfig.from_pretrained(
            self.model_name,
            vocab_size=self.tokenizer.vocab_size,
            num_labels=self.dataset.num_classes,
        )

        self.model = self.get_model_class().from_pretrained(
            self.model_name, config=self.config
        )

        if self.n_layers_to_train > 0:
            layer_acc = 0
            for layer in self.model.base_model.encoder.layer[: -self.n_layers_to_train]:
                print(f"Layer {layer_acc}: OFF")
                for p in layer.parameters():
                    p.requires_grad = False
                layer_acc += 1
            for layer in self.model.base_model.encoder.layer[-self.n_layers_to_train :]:
                print(f"Layer {layer_acc}: ON")
                for p in layer.parameters():
                    p.requires_grad = True
                layer_acc += 1
        else:
            # Turn all grads off except the final classification layer
            for i, p in enumerate(self.model.base_model.parameters()):
                # print(f'Layer {i}: OFF')
                p.requires_grad = False

        print(
            f"multi-classifier initialized; training Classification head and {self.n_layers_to_train} Attention Layers."
        )

        self.training_args = training_args
        self.early_stopping_params = early_stopping_params

        self.train_pct = train_pct

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

        callbacks = []
        if self.early_stopping_params:
            callbacks.append(EarlyStoppingCallback(**self.early_stopping_params))

        if len(callbacks) > 0:
            print("USING EARLY STOPPING.")
        else:
            print("NO EARLY STOPPING.")

        trainer = Trainer(
            model=self.model,
            args=default_training_args,
            data_collator=self.collator,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        return trainer

    def train(self):
        trainer = self.get_trainer()
        trainer.train()

    @staticmethod
    def compute_metrics(pred):
        def save_confusion_matrix(mat):
            try:
                # Make a confusion matrix dir on desktop.
                outdir = Path(
                    os.path.join(
                        os.path.join(os.environ["HOME"], "Desktop"),
                        "confusion_matrices",
                    )
                )
                if not os.path.exists(str(outdir)):
                    os.mkdir(str(outdir))

                now = datetime.now()
                f_string = f"{now.month}-{now.day}-{now.year}_{now.hour}hr-{now.minute}min-{now.second}sec"
                ext = ".pickle"
                fname = outdir / f_string
                fname = str(fname) + ext

                with open(fname, "wb") as f:
                    pickle.dump(mat, f)
                    print(f"Saved confusion matrix to: {fname}")
            except:
                print("Encountered an error logging confusion matrix!")

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1s, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)

        try:
            confusion_matrix_to_log = confusion_matrix(labels, preds)
            save_confusion_matrix(confusion_matrix_to_log)

        except:
            print(
                "compute_metrics broke trying to compute the confusion matrix! Skipping."
            )
            pass

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def get_tokenizer_class(self):
        pass

    @abstractmethod
    def get_config_class(self):
        pass


# TODO: move this to another file?
from transformers import (
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizerFast,
)


class LongformerMultiClassifier(AbstractMultiClassifier):
    def __init__(
        self,
        data_dir,
        model_name="allenai/longformer-base-4096",
        block_size=4096,
        training_args=None,
        ds_limit=np.inf,
        early_stopping_params=None,
        train_pct=0.8,
        data_dir_is_ds=False,
        n_layers_to_train=0,
    ):
        super().__init__(
            data_dir,
            model_name,
            block_size,
            training_args,
            ds_limit,
            early_stopping_params=early_stopping_params,
            train_pct=train_pct,
            data_dir_is_ds=data_dir_is_ds,
            n_layers_to_train=n_layers_to_train,
        )

    def get_model_class(self):
        return LongformerForSequenceClassification

    def get_tokenizer_class(self):
        return LongformerTokenizerFast

    def get_config_class(self):
        return LongformerConfig


# TODO: move this to another file?
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)


class RobertaMultiClassifier(AbstractMultiClassifier):
    def __init__(
        self,
        data_dir,
        model_name="roberta-base",
        block_size=512,
        training_args=None,
        ds_limit=np.inf,
        early_stopping_params=None,
        train_pct=0.8,
        data_dir_is_ds=False,
        n_layers_to_train=0,
    ):
        super().__init__(
            data_dir,
            model_name,
            block_size,
            training_args,
            ds_limit,
            early_stopping_params=early_stopping_params,
            train_pct=train_pct,
            data_dir_is_ds=data_dir_is_ds,
            n_layers_to_train=n_layers_to_train,
        )

    def get_model_class(self):
        return RobertaForSequenceClassification

    def get_tokenizer_class(self):
        return RobertaTokenizerFast

    def get_config_class(self):
        return RobertaConfig
