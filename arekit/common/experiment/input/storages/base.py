import gc
import logging

import numpy as np
import pandas as pd

from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    def __init__(self):
        # TODO. 204. Remove columns provider (now we utilize it as a parameter during filling operation).
        self._columns_provider = None
        self._df = None

    def _create_empty(self):
        # TODO. 204. Pass columns_provider as a parameter.
        data = np.empty(0, dtype=np.dtype(self._columns_provider.get_columns_list_with_types()))
        return pd.DataFrame(data)

    def _balance(self, column_name):
        """ Performs oversampled balancing.
        """
        assert(isinstance(self._df, pd.DataFrame))

        max_size = self._df[column_name].value_counts().max()

        dframes = [self._df]
        for class_index, group in self._df.groupby(column_name):
            dframes.append(group.sample(max_size - len(group), replace=True))

        # Clear resources
        self._df = pd.concat(dframes)
        for df in dframes:
            del df
        gc.collect()

    def set_value(self, row_ind, column, value):
        self._df.at[row_ind, column] = value

    def fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        # TODO. 204. Pass columns_provider as a parameter.
        self._df[self._columns_provider.ROW_ID] = list(range(rows_count))
        self._df.set_index(self._columns_provider.ROW_ID, inplace=True)

    # TODO. 204. Move logic from rows  provider here.
    def fill(self, rows_provider):
        raise NotImplementedError()

    def log_info(self):
        logger.info(self._df.info())

    # region protected methods

    def _dispose_dataframe(self):
        del self._df
        gc.collect()

    # endregion

    # region public methods

    # TODO. 204. Remove.
    def set_columns_provider(self, columns_provider):
        assert(isinstance(columns_provider, BaseColumnsProvider))
        assert(self._columns_provider is None)
        self._columns_provider = columns_provider

    # TODO. 204. Pass columns provider.
    def init_empty(self):
        self._df = self._create_empty()

    # endregion

    # region base methods

    def __iter__(self):
        for row in self._df.iterrows():
            yield row

    def __enter__(self):
        # TODO. 204. Remove (or we should pass columns provider).
        self.init_empty()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dispose_dataframe()

    # endregion
