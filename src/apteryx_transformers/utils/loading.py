import os
import shutil
import subprocess

import torch
from torch.nn import functional as F
import transformers
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    pipeline,
)


def load_pipeline(tokenizer, model_dir, pipeline_mode, redownload=False):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {device}")
    P = pipeline(pipeline_mode, tokenizer=tokenizer, model=model, device=device)
    return P, model, device, model_dir
