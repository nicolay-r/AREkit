import gc
import logging

import numpy as np
import pandas as pd

from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    def __init__(self, columns_provider):
        assert(isinstance(columns_provider, BaseColumnsProvider))
        self._columns_provider = columns_provider
        self._df = None

    def _create_empty(self):
        data = np.empty(0, dtype=np.dtype(self._columns_provider.get_columns_list_with_types()))
        return pd.DataFrame(data)

    def set_value(self, row_ind, column, value):
        self._df.at[row_ind, column] = value

    def fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        self._df[self._columns_provider.ROW_ID] = list(range(rows_count))
        self._df.set_index(self._columns_provider.ROW_ID, inplace=True)

    def log_info(self):
        logger.info(self._df.info())

    # region protected methods

    def _dispose_dataframe(self):
        del self._df
        gc.collect()

    # endregion

    # region public methods

    def init_empty(self):
        self._df = self._create_empty()

    def save(self):
        raise NotImplementedError()

    # endregion

    # region base methods

    def __iter__(self):
        for row in self._df.iterrows():
            yield row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dispose_dataframe()

    # endregion
