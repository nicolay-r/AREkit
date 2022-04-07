import gc
import logging

import numpy as np
import pandas as pd

from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.utils import progress_bar_defined, progress_bar_iter

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    def __init__(self):
        self._df = None

    # region properties

    # TODO. Temporary hack, however this should be removed in future.
    @property
    def DataFrame(self):
        return self._df

    # endregion

    @classmethod
    def from_tsv(cls, filepath, sep='\t', compression='gzip', encoding='utf-8', header="infer", col_types=None):
        instance = cls()

        # Speciall assignation of types for certain columns.
        if col_types is None:
            col_types = dict()

        instance._df = pd.read_csv(filepath,
                                   sep=sep,
                                   encoding=encoding,
                                   compression=compression,
                                   dtype=col_types,
                                   header=header)
        return instance

    # region private methods

    @staticmethod
    def __create_empty(cols_with_types):
        """ cols_with_types: list of pairs ("name", dtype)
        """
        assert(isinstance(cols_with_types, list))
        data = np.empty(0, dtype=np.dtype(cols_with_types))
        return pd.DataFrame(data)

    def __fill_with_blank_rows(self, row_id_column_name, rows_count):
        assert(isinstance(row_id_column_name, str))
        assert(isinstance(rows_count, int))
        self._df[row_id_column_name] = list(range(rows_count))
        self._df.set_index(row_id_column_name, inplace=True)

    def __set_value(self, row_ind, column, value):
        self._df.at[row_ind, column] = value

    def __log_info(self):
        logger.info(self._df.info())

    def __filter(self, column_name, value):
        return self._df[self._df[column_name] == value]

    # endregion

    # region protected methods

    def _balance(self, column_name):
        """ Performs oversampled balancing.
        """
        assert (isinstance(self._df, pd.DataFrame))

        max_size = self._df[column_name].value_counts().max()

        dframes = [self._df]
        for class_index, group in self._df.groupby(column_name):
            dframes.append(group.sample(max_size - len(group), replace=True))

        # Clear resources
        self._df = pd.concat(dframes)
        for df in dframes:
            del df

        gc.collect()

    # endregion

    # region public methods

    def find_by_value(self, column_name, value):
        # TODO. Return new storage. (Encapsulation)
        return self.__filter(column_name=column_name, value=value)

    def find_first_by_value(self, column_name, value):
        # TODO. Return new storage. (Encapsulation)
        row = self.__filter(column_name=column_name, value=value)
        return row.iloc[0]

    def iter_column_values(self, column_name, dtype=None):
        values = self._df[column_name]
        if dtype is None:
            return values
        return values.astype(dtype)

    def get_row(self, row_index):
        return self._df.iloc[row_index]

    def get_cell(self, row_index, column_name):
        return self._df.iloc[row_index][column_name]

    def init_empty(self, columns_provider):
        cols_with_types = columns_provider.get_columns_list_with_types()
        self._df = self.__create_empty(cols_with_types)

    def fill(self, iter_rows_func, columns_provider, desc=""):
        assert(callable(iter_rows_func))
        assert(isinstance(columns_provider, BaseColumnsProvider))

        logged_rows_it = progress_bar_iter(iterable=iter_rows_func(True),
                                           desc="Calculating rows count",
                                           unit="rows")
        rows_count = sum(1 for _ in logged_rows_it)

        logger.info("Filling with blank rows: {}".format(rows_count))
        self.__fill_with_blank_rows(row_id_column_name=columns_provider.ROW_ID,
                                    rows_count=rows_count)
        logger.info("Completed!")

        it = progress_bar_defined(iterable=iter_rows_func(False),
                                  desc="{fmt}".format(fmt=desc),
                                  total=rows_count)

        for row_index, row in enumerate(it):
            for column, value in row.items():
                self.__set_value(row_ind=row_index,
                                 column=column,
                                 value=value)

        self.__log_info()

    def free(self):
        del self._df
        gc.collect()

    @staticmethod
    def __iter_core(df):
        for row_index, row in df.iterrows():
            yield row_index, row

    def iter_shuffled(self):
        shuffled_df = self._df.sample(frac=1)
        return self.__iter_core(shuffled_df)

    # endregion

    # region base methods

    def __iter__(self):
        return self.__iter_core(self._df)

    # endregion
