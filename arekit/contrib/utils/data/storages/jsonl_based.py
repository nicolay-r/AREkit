import json

from arekit.common.data.storages.base import BaseRowsStorage


class JsonlBasedRowsStorage(BaseRowsStorage):

    def __init__(self, rows, **kwargs):
        assert(isinstance(rows, list))
        super(JsonlBasedRowsStorage, self).__init__(**kwargs)
        self.__rows = rows

    def _iter_rows(self):
        for row_index, row in enumerate(self.__rows):
            assert(isinstance(row, str))
            yield row_index, json.loads(row)

    def _get_rows_count(self):
        return len(self.__rows)
