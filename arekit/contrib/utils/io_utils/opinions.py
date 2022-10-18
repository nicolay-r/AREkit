from os.path import join

from arekit.contrib.utils.data.ext import create_reader_extension
from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.contrib.utils.io_utils.utils import filename_template


class OpinionsIO(BaseSamplesIO):

    def __init__(self, target_dir, reader=None, prefix="opinion", target_extension=".tsv.gz"):
        assert(isinstance(reader, BaseReader))
        self.__target_dir = target_dir
        self.__prefix = prefix
        self.__reader = reader
        self.__target_extension = create_reader_extension(reader) \
            if target_extension is None else target_extension

    @property
    def Reader(self):
        return self.__reader

    def create_target(self, data_type, data_folding):
        return self.__get_input_opinions_target(data_type, data_folding=data_folding)

    def __get_input_opinions_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix=self.__prefix,
                                   extension=self.__target_extension)

    @staticmethod
    def __get_filepath(out_dir, template, prefix, extension):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        assert(isinstance(extension, str))
        return join(out_dir, "{prefix}-{template}{extension}".format(
            prefix=prefix, template=template, extension=extension))
