import gzip

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.predict.base_provider import BasePredictProvider


class TsvPredictProvider(BasePredictProvider):

    def __init__(self, filepath):
        assert(isinstance(filepath, str))
        self.__filepath = filepath
        self.__col_separator = '\t'
        self.__f = None

    def __write(self, params):
        self.__f.write("{}\n".format(self.__col_separator.join(params)))

    # region protected

    def _load_header(self, params):
        self.__write(params)

    def _load_content_line(self, params):
        self.__write(params)

    # endregion

    # region base

    def __enter__(self):
        create_dir_if_not_exists(self.__filepath)
        self.__f = gzip.open(self.__filepath, 'wb')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__f.close()

    # endregion
