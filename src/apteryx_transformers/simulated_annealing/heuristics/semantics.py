import numpy as np
import torch
from torch.nn.functional import softmax

from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import logging


class SemanticScorer:
    def __init__(self, model_name='stsb-roberta-large', device='cpu'):
        logger = logging.Logger('Semantic Scorer')

        self.model = SentenceTransformer(model_name).to(device)

    def __call__(self, sequence, target):
        return self.similarity(sequence, target)

    def similarity(self, sequence, target):
        s_emb, t_emb = self.model.encode([sequence, target])
        return (1 / (cosine(s_emb, t_emb)) / np.pi)
