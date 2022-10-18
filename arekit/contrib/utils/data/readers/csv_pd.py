import pandas as pd
from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage


class PandasCsvReader(BaseReader):
    """ Represents a CSV-based reader, implmented via pandas API.
    """

    @staticmethod
    def __from_csv(filepath, sep='\t', compression='gzip',
                   encoding='utf-8', header="infer", col_types=None):

        # Speciall assignation of types for certain columns.
        if col_types is None:
            col_types = dict()

        return pd.read_csv(filepath,
                           sep=sep,
                           encoding=encoding,
                           compression=compression,
                           dtype=col_types,
                           header=header)

    def read(self, target):
        df = PandasCsvReader.__from_csv(filepath=target)
        return PandasBasedRowsStorage(df)
