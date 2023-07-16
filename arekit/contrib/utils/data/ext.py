from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.contrib.utils.data.writers.base import BaseWriter
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter


PANDAS_CSV_EXTENSION = ".tsv.gz"
OPENNRE_EXTENSION = ".jsonl"


def create_writer_extension(writer):
    assert(isinstance(writer, BaseWriter))

    if isinstance(writer, OpenNREJsonWriter):
        return OPENNRE_EXTENSION
    if isinstance(writer, NativeCsvWriter):
        return ".csv"
    else:
        # consider ".tsv.gz" and assuming it is a Pandas.
        return PANDAS_CSV_EXTENSION


def create_reader_extension(writer):
    assert(isinstance(writer, BaseReader))

    if isinstance(writer, OpenNREJsonWriter):
        return OPENNRE_EXTENSION
    else:
        # consider ".tsv.gz" and assuming it is a Pandas.
        # other options are not available in 0.23.1
        return PANDAS_CSV_EXTENSION
