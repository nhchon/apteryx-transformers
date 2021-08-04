import gzip
import os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from pathlib import Path
import dill as pickle
import json


class AutoEncoder_JSONL_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        extract_op,
        tokenizer,
        block_size: int,
        limit=None,
        glob_pattern="*",
    ):
        self.block_size = block_size
        self.tok = tokenizer
        self.DS_LIMIT = limit
        self.BLOCK_SIZE = block_size
        print("Ingesting data!")
        self.data = self.get_data(data_dir, extract_op, limit)

    def __len__(self):
        return self.data.labels.shape[0]

    def __getitem__(self, item):
        return {
            "input_ids": self.data.input_ids[item],
            "attention_mask": self.data.attention_mask[item],
            "labels": self.data.labels[item],
        }

    def get_data(self, data_dir, extract_op, limit):
        files = Path(data_dir).glob("*")

        jdata = []
        acc = 0
        for file_n, file in enumerate(files):
            print(f"File {file_n}: {file}")
            print(f"{acc} processed so far.")
            with open(file, "r") as f:
                for line in tqdm(f.readlines()):
                    if acc < self.DS_LIMIT:
                        l = extract_op(json.loads(line))

                        enc = self.tok(l, return_tensors="pt")
                        ids = enc.input_ids

                        i_len = ids.shape[-1]
                        n_blocks = i_len // self.BLOCK_SIZE
                        trunc_len = n_blocks * self.BLOCK_SIZE

                        # Chunked ids
                        jdata.extend(
                            self.tok.batch_decode(
                                (ids[:, :trunc_len].view(n_blocks, self.BLOCK_SIZE))
                            )
                        )

                        acc += n_blocks
                    else:
                        tokenized = self.tok(
                            jdata,
                            padding="max_length",
                            truncation=True,
                            max_length=self.block_size,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )

                        # Add labels for autoencoder!
                        tokenized.update({"labels": tokenized.input_ids})

                        return tokenized


class BalancedDataset(Dataset):
    def __init__(self, tokenizer, data, block_size: int, limit=None):
        self.block_size = block_size
        self.tok = tokenizer
        print("Ingesting data!")
        self.txt = [i[0] for i in tqdm(data[:limit])]
        self.labels = torch.tensor([i[1] for i in tqdm(data[:limit])])

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, item):
        d = self.tok(
            self.txt[item],
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
            add_special_tokens=False,
        )
        d["labels"] = self.labels[item]
        return d


class PickleDataset(Dataset):
    def __init__(self, data_dir, tokenizer, block_size, limit=None):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.txt = []
        self.labels = []
        print(f"Ingesting Data from {data_dir}!")
        pickle_files = list(Path(data_dir).glob("*.pickle"))
        for f in tqdm(pickle_files[:limit]):
            with open(f, "rb") as data:
                chunks = pickle.load(data)
                for t, l in chunks:
                    self.txt.append(t)
                    self.labels.append(l)
        self.labels = torch.Tensor(self.labels).long()

    @property
    def num_classes(self):
        return len(self.labels.unique())

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, item):
        d = self.tokenizer(
            self.txt[item],
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
            add_special_tokens=False,
        )
        d["labels"] = self.labels[item]
        return d


# load from disk/pkl
# tokenize, get ids, chunk in block size, etc
# pad last chunk?
class ExpandedDataset(Dataset):
    # dataset: list of patent objects
    def __init__(self, tokenizer, dataset, class_labels, block_size: int, limit=None):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.class_map = {x: i for i, x in enumerate(class_labels)}
        print("Ingesting data!")
        self.labels = list()
        self.input_ids = list()
        for item in tqdm(dataset[:limit]):
            # 'publication_number', 'TC', 'pos', 'abs_text', 'desc_text', 'claims_text'
            item_txt = "\n".join(
                [item["abs_text"], item["desc_text"], item["claims_text"]]
            )
            tokenized_txt = self.tokenizer(
                item_txt, return_tensors="pt"
            )  # input_ids, attention_mask
            input_ids = tokenized_txt["input_ids"]
            q_tokens = input_ids.shape[1]
            # print('tokens in pat:', q_tokens)
            chunks = [
                input_ids[0, i : i + block_size] for i in range(0, q_tokens, block_size)
            ]
            for chunk in chunks:
                # print(chunk.shape)
                self.input_ids.append(chunk)
            # print('q chunks', len(chunks))
            label = self.class_map[int(item["TC"])]
            for i in range(len(chunks)):
                self.labels.append(label)

        self.labels = torch.tensor(self.labels)

    @property
    def num_classes(self):
        return len(self.labels.unique())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        import torch.nn.functional as F

        input_ids = self.input_ids[item]
        mask = torch.ones(self.block_size)
        if len(input_ids) < self.block_size:
            delta = self.block_size - len(input_ids)
            input_ids = F.pad(
                input_ids,
                [0, delta],
                mode="constant",
                value=self.tokenizer.pad_token_id,
            )
            mask[-delta:] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": self.labels[item],
        }


class PickleDatasetFromDisk(Dataset):
    def __init__(self, data_dir, tokenizer, block_size, limit=np.inf):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        print("Indexing dir...")
        self.index_map = dict()
        for i, p in tqdm(enumerate(Path(data_dir).glob("*/*.pickle"))):
            if i < limit:
                self.index_map[i] = p
            else:
                break

    @property
    def num_classes(self):
        return 8

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        # Map int item to file path
        chunk_file = self.index_map[item]
        with open(chunk_file, "rb") as f:
            txt, label = pickle.load(f)

        label = torch.Tensor([label]).long()
        d = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
            add_special_tokens=False,
        )

        d["labels"] = label

        return d


class PickleDatasetByClass(Dataset):
    def __init__(
        self,
        index_path,
        data_dir,
        tokenizer,
        block_size,
        per_class_limit,
        is_autoencoder=False,
    ):
        assert "by_class" in str(index_path), "Not using a _by_class dataset!"
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.index_path = index_path
        self.data_dir = data_dir
        self.is_autoencoder = is_autoencoder
        print("Loading index...")
        with open(self.index_path, "rb") as f:
            self.class_to_patent_map = pickle.load(f)
        self.num_classes = len(self.class_to_patent_map)
        self.per_class_limit = per_class_limit

        print("Randomizing and truncating data...")
        for _class in tqdm(range(self.num_classes)):
            # Make sure no class has too few examples
            assert (
                len(self.class_to_patent_map[_class]) > self.per_class_limit
            ), f"Make sure per_class_limit is less than {self.class_to_patent_map[_class]}."

            self.class_to_patent_map[_class] = np.random.choice(
                self.class_to_patent_map[_class],
                size=self.per_class_limit,
                replace=False,
            )

    def __len__(self):
        return self.num_classes * self.per_class_limit

    def __getitem__(self, item):
        _class = item // self.per_class_limit
        _class_idx = item % self.per_class_limit

        # Map int item to file path
        chunk_file = os.path.join(
            self.data_dir, self.class_to_patent_map[_class][_class_idx]
        )
        with open(chunk_file, "rb") as f:
            txt, label = pickle.load(f)
            assert int(_class) == int(
                label
            ), f"Label {label} doesn't match Class {_class}!"

        label = torch.Tensor([label]).long()
        d = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
            add_special_tokens=False,
        )

        d["labels"] = d.input_ids if self.is_autoencoder else label

        return d


class GzippedPickleDatasetByClass(PickleDatasetByClass):
    def __init__(
        self,
        gz_index_path,
        data_dir,
        tokenizer,
        block_size,
        per_class_limit,
        absolute_limit=None,
    ):
        assert "by_class" in str(gz_index_path), "Not using a _by_class dataset!"
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.gz_index_path = gz_index_path
        self.data_dir = data_dir
        print("Loading index...")
        with gzip.open(self.gz_index_path, "rb") as f:
            self.class_to_patent_map = pickle.load(f)
        self.num_classes = len(self.class_to_patent_map)

        if absolute_limit:
            print("absolute_limit used - overriding per_class_limit if provided.")
            self.per_class_limit = int(absolute_limit / self.num_classes)
        else:
            self.per_class_limit = per_class_limit

        print("Randomizing and truncating data...")
        for _class in tqdm(range(self.num_classes)):
            # Make sure no class has too few examples
            assert (
                len(self.class_to_patent_map[_class]) > self.per_class_limit
            ), f"Make sure per_class_limit is less than {self.class_to_patent_map[_class]}."

            self.class_to_patent_map[_class] = np.random.choice(
                self.class_to_patent_map[_class],
                size=self.per_class_limit,
                replace=False,
            )
