import gc
import logging

import numpy as np
import pandas as pd

from arekit.common.experiment.data_type import DataType

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    ROW_ID = 'row_id'

    def __init__(self, data_type):
        assert(isinstance(data_type, DataType))
        self._data_type = data_type
        self._df = self._create_empty_df()

    # region protected methods

    def _get_columns_list_with_types(self):
        dtypes_list = list()
        dtypes_list.append((BaseRowsStorage.ROW_ID, 'int32'))
        return dtypes_list

    def _create_empty_df(self):
        data = np.empty(0, dtype=np.dtype(self._get_columns_list_with_types()))
        return pd.DataFrame(data)

    # endregion

    # region private methods

    def set_value(self, row_ind, column, value):
        self._df.at[row_ind, column] = value

    def fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        self._df[self.ROW_ID] = list(range(rows_count))
        self._df.set_index(self.ROW_ID, inplace=True)

    def log_info(self):
        logger.info(self._df.info())

    # endregion

    def __iter__(self):
        for row in self._df.iterrows():
            yield row

    def _dispose_dataframe(self):
        del self._df
        gc.collect()

    def save(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dispose_dataframe()
