import os
import numpy as np


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
