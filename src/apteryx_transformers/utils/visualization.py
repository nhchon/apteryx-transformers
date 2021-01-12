import json
from tqdm import tqdm

import torch
from transformers import TrainingArguments
from transformers import Trainer

from .smooth_gradient import SmoothGradient

from ..datasets import BalancedDataset
from ..collators import DataCollatorForDocumentClassificationBATCH

# ----------------------------------------------------------------------------------------------------------------------------------


class Visualizer:
    def __init__(self, model, tokenizer, window_size, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.base_output_dir = output_dir
        self.criterion = torch.nn.CrossEntropyLoss()
        self.collator = DataCollatorForDocumentClassificationBATCH()

    def visualize(self, txt, threshold=.7):
        '''
        :param txt: Document text. A string.
        :param threshold: Percentage of significance to report.
        :return: A colored HTML string.
        '''
        label = -1 #Only needed for underlying datashape. Should always be -1.

        ids = self.tokenizer(txt, add_special_tokens=False)['input_ids']
        chunked_data = [(self.tokenizer.decode(ids[i:i + self.window_size]), label)
                        for i in range(0, len(ids), self.window_size)]
        ds = BalancedDataset(self.tokenizer, chunked_data, block_size=self.window_size)
        training_args = TrainingArguments(
            output_dir=self.base_output_dir,  # TODO N: generate a better directory
            per_device_train_batch_size=2,
            local_rank=-1)
        trainer = Trainer(model=self.model,
                          args=training_args,
                          data_collator=self.collator,
                          train_dataset=ds)
        smooth_grad = SmoothGradient(self.model,
                                     self.criterion,
                                     self.tokenizer,
                                     show_progress=True,
                                     encoder="bert")
        instances = smooth_grad.saliency_interpret(trainer.get_train_dataloader())
        result = list()
        for i in instances:
            if i['prob'] > threshold:
                colored_string = smooth_grad.colorize(i)
                result.append(colored_string)
                # TODO 34: do something with this
        return result
