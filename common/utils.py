import collections
import os
import numpy as np

from arekit.common.bound import Bound


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


def iter_text_with_substitutions(text, iter_subs):
    """
    text: list or string
        where we perform substitutions;
        list -- list of terms
        string -- list of chars
    iter_subs: Iterable of (value, Bound) pairs

    NOTE: substitutions should be ordered!
    """
    assert(isinstance(text, list) or isinstance(text, unicode))
    assert(isinstance(iter_subs, collections.Iterable))

    start = 0

    is_list = False
    if isinstance(text, list):
        is_list = True

    for value, bound in iter_subs:
        assert(isinstance(bound, Bound))
        assert(bound.Position >= start)

        for part in __iter_text_part(text_part=text[start:bound.Position], is_list=is_list):
            yield part

        yield value

        start = bound.Position + bound.Length

    for part in __iter_text_part(text_part=text[start:len(text)], is_list=is_list):
        yield part


def __iter_text_part(text_part, is_list):
    if is_list:
        for word in text_part:
            yield word
    else:
        yield text_part
