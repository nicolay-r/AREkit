import gc
import logging

from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.utils import progress_bar_defined

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    # region abstract methods

    def _set_value(self, row_ind, column, value):
        raise NotImplemented()

    def _iter_rows(self):
        raise NotImplemented()

    def _get_rows_count(self):
        raise NotImplemented()

    def find_by_value(self, column_name, value):
        raise NotImplemented()

    def find_first_by_value(self, column_name, value):
        raise NotImplemented()

    def iter_column_values(self, column_name, dtype=None):
        raise NotImplemented()

    def get_row(self, row_index):
        raise NotImplemented()

    def get_cell(self, row_index, column_name):
        raise NotImplemented()

    def init_empty(self, columns_provider):
        raise NotImplemented()

    def iter_shuffled(self):
        raise NotImplemented()

    # endregion

    def fill(self, iter_rows_func, columns_provider, rows_count=None, desc=""):
        assert(callable(iter_rows_func))
        assert(isinstance(columns_provider, BaseColumnsProvider))

        it = progress_bar_defined(iterable=iter_rows_func(False),
                                  desc="{fmt}".format(fmt=desc),
                                  total=rows_count)

        for row_index, row in enumerate(it):
            for column, value in row.items():
                self._set_value(row_ind=row_index,
                                column=column,
                                value=value)

    def free(self):
        gc.collect()

    # endregion

    # region base methods

    def __iter__(self):
        return self._iter_rows()

    def __len__(self):
        return self._get_rows_count()

    # endregion
