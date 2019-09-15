import collections
import os
import numpy as np

from core.common.bound import Bound


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

        __iter_text_part(text_part=text[start:bound.Position - start],
                         is_list=is_list)

        yield value

        start = bound.Position + bound.Length

    yield __iter_text_part(text_part=text[start:len(text) - start],
                           is_list=is_list)


def __iter_text_part(text_part, is_list):
    if is_list:
        for word in text_part:
            yield word
    else:
        yield text_part
