import gzip

from arekit.common.utils import create_dir_if_not_exists, progress_bar_iter


class TsvPredictWriter(object):

    def __init__(self, filepath):
        assert(isinstance(filepath, str))
        self.__filepath = filepath
        self.__col_separator = '\t'
        self.__f = None

    def __write(self, params):
        self.__f.provide("{}\n".format(self.__col_separator.join(params)))

    def write(self, title, contents_it):
        self.__write(title)

        wrapped_it = progress_bar_iter(iterable=contents_it,
                                       desc='Writing output',
                                       unit='rows')

        for contents in wrapped_it:
            self.__write(contents)

    # region base

    def __enter__(self):
        create_dir_if_not_exists(self.__filepath)
        self.__f = gzip.open(self.__filepath, 'wb')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__f.close()

    # endregion
