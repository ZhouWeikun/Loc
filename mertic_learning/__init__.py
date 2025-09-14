from .base_distance import BaseDistance
from .lp_distance import LpDistance
from .dot_product_similarity import DotProductSimilarity

import torch
def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min