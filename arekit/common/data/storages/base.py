import gc
import logging

from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.utils import progress_bar

logger = logging.getLogger(__name__)


class BaseRowsStorage(object):

    # region protected methods

    def _begin_filling_row(self, row_ind):
        pass

    # endregion

    # region abstract methods

    def _set_row_value(self, row_ind, column, value):
        raise NotImplemented()

    def _iter_rows(self):
        """ returns: tuple(int, list)
                provides the index (int) and the related content of the row (dict)
        """
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

    def iter_column_names(self):
        raise NotImplemented()

    # endregion

    def fill(self, iter_rows_func, columns_provider, row_handler=None, rows_count=None, desc=""):
        assert(callable(iter_rows_func))
        assert(isinstance(columns_provider, BaseColumnsProvider))
        assert(callable(row_handler) or row_handler is None)

        pbar_it = progress_bar(iterable=iter_rows_func(False),
                               desc="{fmt}".format(fmt=desc),
                               total=rows_count)

        doc_ids_seen = set()

        for row_index, row in enumerate(pbar_it):

            doc_id, row_values = row

            self._begin_filling_row(row_index)

            for column, value in row_values.items():
                self._set_row_value(row_ind=row_index,
                                    column=column,
                                    value=value)

            # Provide information about amount of processed documents.
            doc_ids_seen.add(doc_id)
            pbar_it.set_postfix({
                "docs_seen": len(doc_ids_seen),
                "doc_now": doc_id,
            })

            if row_handler is not None:
                row_handler()

    def free(self):
        gc.collect()

    # endregion

    # region base methods

    def __iter__(self):
        return self._iter_rows()

    def __len__(self):
        return self._get_rows_count()

    # endregion
