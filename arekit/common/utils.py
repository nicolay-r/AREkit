from os import makedirs
from os.path import dirname, exists
from tqdm import tqdm


def create_dir_if_not_exists(filepath):
    dir = dirname(filepath)

    # Check whether string is empty.
    if not dir:
        return

    if not exists(dir):
        makedirs(dir)


def filter_whitespaces(terms):
    return [term.strip() for term in terms if term.strip()]


def split_by_whitespaces(text):
    """
    Assumes to perform a word separation including a variety of space entries.
    In terms of the latter we consider any whitespace separator.
    """
    assert(isinstance(text, str))
    return text.split()


def progress_bar_defined(iterable, total, desc="", unit="it"):
    return tqdm(iterable=iterable,
                total=total,
                desc=desc,
                ncols=120,
                position=0,
                leave=True,
                unit=unit,
                miniters=total / 200)


def progress_bar_iter(iterable, desc="", unit='it'):
    return tqdm(iterable=iterable,
                desc=desc,
                position=0,
                leave=True,
                ncols=120,
                unit=unit)
