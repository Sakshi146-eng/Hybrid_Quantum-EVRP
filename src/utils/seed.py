"""
Reproducibility utilities for setting random seeds across libraries.
"""
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Sets global seed for Python random and NumPy for experiment reproducibility.

    Args:
        seed (int): Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seed set conditionally if imported
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
