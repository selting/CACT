import hashlib
import random

import numpy as np
import torch

rng = None


def set_all_seeds(seed=42, loader=None):
    """Sets all seeds (python's random module, numpy.random, torch, cuda) to a given value"""
    random.seed(seed)
    np.random.seed(seed)
    global rng
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if loader is not None:
        loader.sampler.generator.manual_seed(seed)
    pass


def generate_unique_seed(*args):
    # Convert all arguments to their string representations
    arg_strings = [str(arg) for arg in args]

    # Concatenate the string representations
    seed_string = ''.join(arg_strings)

    # Generate a hash value using SHA256 algorithm
    hash_object = hashlib.sha256(seed_string.encode())
    seed = int(hash_object.hexdigest(), 16)

    # Adjust seed within the valid range for np.random.seed()
    seed %= 2 ** 32

    return seed
