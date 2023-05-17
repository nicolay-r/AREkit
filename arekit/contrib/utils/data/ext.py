from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.writers.base import BaseWriter
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter


PANDAS_CSV_EXTENSION = ".tsv.gz"
OPENNRE_EXTENSION = ".jsonl"


def create_writer_extension(writer):
    assert(isinstance(writer, BaseWriter))

    if isinstance(writer, OpenNREJsonWriter):
        return OPENNRE_EXTENSION
    if isinstance(writer, PandasCsvWriter):
        return PANDAS_CSV_EXTENSION
    if isinstance(writer, NativeCsvWriter):
        return ".csv"

    raise NotImplementedError()


def create_reader_extension(writer):
    assert(isinstance(writer, BaseReader))

    if isinstance(writer, PandasCsvReader):
        return PANDAS_CSV_EXTENSION
    if isinstance(writer, OpenNREJsonWriter):
        return OPENNRE_EXTENSION

    raise NotImplementedError()
