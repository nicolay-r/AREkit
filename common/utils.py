import os
import numpy as np
from tqdm import tqdm


def get_random_uniform_with_fixed_seed(vector_size, seed):
    """
    Generates random vector by specific initial 'seed' value
    """
    assert(isinstance(vector_size, int))
    assert(isinstance(seed, int))
    np.random.seed(seed)
    return np.random.uniform(-1, 1, vector_size)


def get_random_normal_distribution(vector_size, seed, loc, scale):
    assert(isinstance(vector_size, int))
    assert(isinstance(seed, int))
    np.random.seed(seed)
    return np.random.normal(loc=loc, scale=scale, size=vector_size)


def create_dir_if_not_exists(filepath):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)


def filter_whitespaces(terms):
    return [term.strip() for term in terms if term.strip()]


def split_by_whitespaces(text):
    """
    Assumes to perform a word separation including a variety of space entries.
    In terms of the latter we consider any whitespace separator.
    """
    assert(isinstance(text, unicode))
    return text.split()


def progress_bar_defined(iterable, total, desc="", unit="it"):
    return tqdm(iterable=iterable,
                total=total,
                desc=desc,
                ncols=80,
                position=0,
                leave=True,
                unit=unit,
                miniters=total / 200)


def progress_bar_iter(iterable, desc="", unit='it'):
    return tqdm(iterable=iterable,
                desc=desc,
                position=0,
                leave=True,
                ncols=80,
                unit=unit)
