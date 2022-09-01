from arekit.common.data.input.writers.opennre_json import OpenNREJsonWriter
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.input.writers.base import BaseWriter


def create_writer_extension(writer):
    assert(isinstance(writer, BaseWriter))

    if isinstance(writer, OpenNREJsonWriter):
        return ".json"
    if isinstance(writer, TsvWriter):
        return ".tsv.gz"

    raise NotImplementedError()
