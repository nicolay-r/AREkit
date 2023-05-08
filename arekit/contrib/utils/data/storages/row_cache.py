from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.data.storages.base import BaseRowsStorage


class RowCacheStorage(BaseRowsStorage):
    """ Row Caching storage kernel, based on python dictionary.
    """

    def __init__(self):
        self.__f = None
        self.__row_cache = {}
        self.__columns = []

    @property
    def RowCache(self):
        return self.__row_cache

    def init_empty(self, columns_provider):
        assert(isinstance(columns_provider, BaseColumnsProvider))
        for col_name, _ in columns_provider.get_columns_list_with_types():
            self.__columns.append(col_name)

    def iter_column_names(self):
        return iter(self.__columns)

    def _set_row_value(self, row_ind, column, value):
        self.__row_cache[column] = value

    def _begin_filling_row(self, row_ind):
        self.__row_cache.clear()
