import torch
import logging


class LengthScorer:
    def __init__(self):
        logger = logging.Logger("Fluency Scorer")

    def __call__(self, source, target, c=1):
        return c * min(len(source), len(target)) / max(len(source), len(target))
