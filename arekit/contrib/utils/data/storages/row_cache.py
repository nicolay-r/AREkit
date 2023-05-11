from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.data.storages.base import BaseRowsStorage


class RowCacheStorage(BaseRowsStorage):
    """ Row Caching storage kernel, based on python dictionary.
    """

    def __init__(self, force_collect_columns=None):
        """ This is a particular/related solution for the following issue:
            https://github.com/nicolay-r/AREkit/issues/464
            force_collect_columns: list
                columns that supposed to be additionally considered in output.
        """
        assert(isinstance(force_collect_columns, list) or force_collect_columns is None)
        self.__f = None
        self.__row_cache = {}
        self.__columns = []
        self.__force_collect_columns = [] if force_collect_columns is None else force_collect_columns

    @property
    def RowCache(self):
        return self.__row_cache

    def init_empty(self, columns_provider):
        assert (isinstance(columns_provider, BaseColumnsProvider))
        for col_name, _ in columns_provider.get_columns_list_with_types():
            self.__columns.append(col_name)

        # Expand with columns that are forced to be provided.
        existed_set = set(self.__columns)
        self.__columns += [c for c in self.__force_collect_columns if c not in existed_set]

    def iter_column_names(self):
        return iter(self.__columns)

    def _set_row_value(self, row_ind, column, value):
        self.__row_cache[column] = value

    def _begin_filling_row(self, row_ind):
        self.__row_cache.clear()
