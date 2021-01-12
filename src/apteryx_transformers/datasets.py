import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class BalancedDataset(Dataset):
    def __init__(self, tokenizer, data, block_size: int, limit=None):
        self.block_size = block_size
        self.tok = tokenizer
        print('Ingesting data!')
        self.txt = [i[0] for i in tqdm(data[:limit])]
        self.labels = torch.tensor([i[1] for i in tqdm(data[:limit])])

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, item):
        d = self.tok(self.txt[item],
                     padding='max_length',
                     truncation=True,
                     max_length=self.block_size,
                     return_tensors='pt',
                     add_special_tokens = False)
        d['labels'] = self.labels[item]
        return d