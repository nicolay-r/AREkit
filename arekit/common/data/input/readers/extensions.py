from arekit.common.data.input.readers.base import BaseReader
from arekit.common.data.input.readers.tsv import TsvReader


def create_reader_extension(writer):
    assert(isinstance(writer, BaseReader))

    if isinstance(writer, TsvReader):
        return ".tsv.gz"

    raise NotImplementedError()
