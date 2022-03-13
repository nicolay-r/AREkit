from enum import Enum
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


class EnumConversionService(object):

    _data = None

    @classmethod
    def is_supported(cls, name):
        assert(isinstance(cls._data, dict))
        return name in cls._data

    @classmethod
    def name_to_type(cls, name):
        assert(isinstance(cls._data, dict))
        assert(isinstance(name, str))
        return cls._data[name]

    @classmethod
    def iter_names(cls):
        assert(isinstance(cls._data, dict))
        return iter(list(cls._data.keys()))

    @classmethod
    def type_to_name(cls, enum_type):
        assert(isinstance(cls._data, dict))
        assert(isinstance(enum_type, Enum))

        for item_name, item_type in cls._data.items():
            if item_type == enum_type:
                return item_name

        raise NotImplemented("Formatting type '{}' does not supported".format(enum_type))
