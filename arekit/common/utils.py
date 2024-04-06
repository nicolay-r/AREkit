import os
from tqdm import tqdm


def create_dir_if_not_exists(filepath):
    dir = os.path.dirname(filepath)

    # Check whether string is empty.
    if not dir:
        return

    if not os.path.exists(dir):
        os.makedirs(dir)


def filter_whitespaces(terms):
    return [term.strip() for term in terms if term.strip()]


def split_by_whitespaces(text):
    """
    Assumes to perform a word separation including a variety of space entries.
    In terms of the latter we consider any whitespace separator.
    """
    assert(isinstance(text, str))
    return text.split()


def progress_bar(iterable, total, desc="", unit="it", file=None, disable=False):
    if total is not None:
        return progress_bar_defined(iterable=iterable, total=total, desc=desc, unit=unit, file=file, disable=disable)
    else:
        return progress_bar_iter(iterable=iterable, desc=desc, unit=unit, file=file, disable=disable)


def progress_bar_conditional(iterable, condition_func, total, postfix_func=None, desc="", unit="it", file=None):
    """ This progress-bar updates only on the
        specific conditions during the iteration process.
    """
    assert(callable(condition_func))
    assert(callable(postfix_func) or postfix_func is None)

    # We consider artificial function that always iters 0.
    def __iter_infinite_placeholder():
        while True:
            yield 0

    pbar_it = progress_bar(iterable=__iter_infinite_placeholder(),
                           desc=desc, unit=unit, total=total, file=file)
    element = iter(pbar_it)

    # Initialize with 0.
    next(element)

    for item in iterable:

        # Optionally Update progress bar with the next state.
        if condition_func(item):
            next(element)
            yield item

        # Optionally provide meta-information.
        if postfix_func is not None:
            pbar_it.set_postfix(postfix_func(item))


def progress_bar_defined(iterable, total, miniters=200, desc="", unit="it", file=None, disable=False):
    return tqdm(iterable=iterable,
                total=total,
                desc=desc,
                ncols=120,
                position=0,
                leave=True,
                unit=unit,
                file=file,
                disable=disable,
                miniters=total / miniters if total is not None else total)


def progress_bar_iter(iterable, desc="", unit='it', file=None, disable=False):
    return tqdm(iterable=iterable,
                desc=desc,
                position=0,
                leave=True,
                ncols=120,
                file=file,
                disable=disable,
                unit=unit)
