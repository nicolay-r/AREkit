import sqlite3
from arekit.common.data.storages.base import BaseRowsStorage


class SQliteBasedRowsStorage(BaseRowsStorage):

    def __init__(self, path, table_name):
        self.__path = path
        self.__table_name = table_name
        self.__conn = None

    def _iter_rows(self):
        with sqlite3.connect(self.__path) as conn:
            cursor = conn.execute(f"select * from {self.__table_name}")
            for row_index, row in enumerate(cursor.fetchall()):
                row_dict = {cursor.description[i][0]: value for i, value in enumerate(row)}
                yield row_index, row_dict
