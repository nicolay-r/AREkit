import logging

from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.contrib.utils.data.writers.base import BaseWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SamplesIO(BaseSamplesIO):
    """ Samples default IO utils for samples.
            Sample is a text part which include pair of attitude participants.
            This class allows to provide saver and loader for such entries, dubbed as samples.
            Samples required for machine learning training/inferring.
    """

    def __init__(self, create_target_func, writer=None, reader=None):
        assert(isinstance(writer, BaseWriter) or writer is None)
        assert(isinstance(reader, BaseReader) or reader is None)
        assert(callable(create_target_func))

        self.__writer = writer
        self.__reader = reader
        self.__create_target_func = create_target_func

        self.__target_extension = None
        if writer is not None:
            self.__target_extension = writer.extension()
        elif reader is not None:
            self.__target_extension = reader.extension()

    @property
    def Reader(self):
        return self.__reader

    @property
    def Writer(self):
        return self.__writer

    def create_target(self, data_type):
        return self.__create_target_func(data_type) + self.__target_extension
