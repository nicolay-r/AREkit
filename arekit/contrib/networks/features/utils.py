import numpy as np


def in_window(window_begin, window_end, ind):
    return window_begin <= ind < window_end


def pad_right_inplace(lst, pad_size, filler):

    assert(pad_size - len(lst) > 0)
    lst.extend([filler] * (pad_size - len(lst)))


def pad_right_or_crop_inplace(lst, pad_size, filler):

    if len(lst) < pad_size:
        pad_right_inplace(lst, pad_size, filler)
    else:
        del lst[pad_size:]


def create_filled_array(size, value):
    return np.full(size, value, dtype=np.int32)


def create_zeros(size):
    return np.zeros(size, dtype=np.int32)

