def pad_right_inplace(lst, pad_size, filler):

    assert(pad_size - len(lst) > 0)
    lst.extend([filler] * (pad_size - len(lst)))


def in_window(window_begin, window_end, ind):
    return window_begin <= ind < window_end
