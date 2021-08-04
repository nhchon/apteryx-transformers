import json
from tqdm import tqdm

import torch
from transformers import TrainingArguments
from transformers import Trainer

from .smooth_gradient import SmoothGradient

from ..apteryx_datasets import BalancedDataset
from ..collators import DataCollatorForDocumentClassificationBATCH

# ----------------------------------------------------------------------------------------------------------------------------------


class Visualizer:
    def __init__(self, model, tokenizer, window_size, class_map: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.base_output_dir = "./tmp"
        self.criterion = torch.nn.CrossEntropyLoss()
        self.collator = DataCollatorForDocumentClassificationBATCH()
        self.class_map = class_map

    def parse_relative_location_from_response(self, i, txt, offset):
        tok_probs = list(zip(i["tokens"], i["grad"]))

        tmp = txt
        acc = 0
        start_end = []
        for tok, prob in tok_probs:
            tok = tok.replace("Ġ", "")
            tok = tok.replace("Ċ", "")
            start = tmp.index(tok)
            actual_start = acc + start
            end = start + len(tok)
            actual_end = acc + end
            tmp = tmp[end:]

            # print(tok, txt[actual_start:actual_end], start, end)
            start_end.append(
                [
                    tok,
                    txt[actual_start:actual_end],
                    prob,
                    actual_start + offset,
                    actual_end + offset,
                ]
            )

            acc += end

        return start_end

    def visualize(self, txt, threshold=0.7):
        """
        :param txt: Document text. A string.
        :param threshold: Percentage of significance to report.
        :return: A colored HTML string.
        """
        label = -1  # Only needed for underlying datashape. Should always be -1.

        ids = self.tokenizer(txt, add_special_tokens=False)["input_ids"]
        chunked_data = [
            (self.tokenizer.decode(ids[i : i + self.window_size]), label)
            for i in range(0, len(ids), self.window_size)
        ]
        ds = BalancedDataset(self.tokenizer, chunked_data, block_size=self.window_size)
        training_args = TrainingArguments(
            output_dir=self.base_output_dir,  # TODO N: generate a better directory
            per_device_train_batch_size=2,
            local_rank=-1,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collator,
            train_dataset=ds,
        )
        smooth_grad = SmoothGradient(
            self.model,
            self.criterion,
            self.tokenizer,
            show_progress=True,
            encoder="bert",
        )
        instances = smooth_grad.saliency_interpret(trainer.get_train_dataloader())

        start_end_data = []
        chunk_len_acc = 0
        for idx, instance in enumerate(instances):
            try:
                chunk = chunked_data[idx][0]
                loc_data = self.parse_relative_location_from_response(
                    instance, chunk, offset=chunk_len_acc
                )
                start_end_data.append(loc_data)
            except:
                continue
            chunk_len_acc += len(chunk)

        results = list()
        for i in instances:
            if i["prob"] > threshold:
                colored_string = smooth_grad.colorize(i, self.class_map)
                results.append(colored_string)
                # TODO 34: do something with this
        return results, instances, start_end_data
