from dataclasses import dataclass
from typing import Dict, List, Union

import torch


def ensure_correct_dims(t, batch_size):
    n_seq = int(t.flatten().shape[0] / batch_size)
    return t.reshape((batch_size, n_seq))


@dataclass
class DataCollatorForDocumentClassificationBATCH:
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        return {
            "input_ids": ensure_correct_dims(
                torch.stack([e["input_ids"] for e in examples], dim=0), batch_size
            ),
            "attention_mask": ensure_correct_dims(
                torch.stack([e["attention_mask"] for e in examples], dim=0), batch_size
            ),
            "labels": ensure_correct_dims(
                torch.stack([e["labels"] for e in examples], dim=0), batch_size
            ),
        }


@dataclass
class DataCollatorForAutoencodersBATCH:
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        return {
            "input_ids": ensure_correct_dims(
                torch.stack([e["input_ids"] for e in examples], dim=0), batch_size
            ),
            "attention_mask": ensure_correct_dims(
                torch.stack([e["attention_mask"] for e in examples], dim=0), batch_size
            ),
            "labels": ensure_correct_dims(
                torch.stack([e["labels"] for e in examples], dim=0), batch_size
            ),
        }
