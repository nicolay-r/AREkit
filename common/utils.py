import os
import numpy as np


def get_random_vector(vector_size, seed):
    """
    Generates random vector by specific initial 'seed' value
    """
    assert(isinstance(vector_size, int))
    assert(isinstance(seed, int))
    state = np.random.RandomState(seed)
    return state.random_sample(vector_size)

def create_dir_if_not_exists(filepath):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)
