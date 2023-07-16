import importlib
from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage


class PandasCsvReader(BaseReader):
    """ Represents a CSV-based reader, implemented via pandas API.
    """

    def __init__(self, sep='\t', header='infer', compression='infer', encoding='utf-8', col_types=None):
        self.__sep = sep
        self.__compression = compression
        self.__encoding = encoding
        self.__header = header

        # Speciall assignation of types for certain columns.
        self.__col_types = col_types
        if self.__col_types is None:
            self.__col_types = dict()

    def __from_csv(self, filepath):
        pd = importlib.import_module("pandas")
        return pd.read_csv(filepath,
                           sep=self.__sep,
                           encoding=self.__encoding,
                           compression=self.__compression,
                           dtype=self.__col_types,
                           header=self.__header)

    def read(self, target):
        df = self.__from_csv(filepath=target)
        return PandasBasedRowsStorage(df)
