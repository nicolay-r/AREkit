import os
import sqlite3
from os.path import dirname

from arekit.common.data import const
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter


class SQliteWriter(BaseWriter):

    def __init__(self, table_name="contents", index_column_names=None, skip_existed=False, clear_table=True):
        """ index_column_names: list or None
                column names should be considered to build a unique index;
                if None, the default 'const.ID' will be considered for row indexation.
        """
        assert (isinstance(index_column_names, list) or index_column_names is None)
        self.__index_column_names = index_column_names if index_column_names is not None else [const.ID]
        self.__table_name = table_name
        self.__conn = None
        self.__cur = None
        self.__need_init_table = True
        self.__origin_column_names = None
        self.__skip_existed = skip_existed
        self.__clear_table = clear_table

    def extension(self):
        return ".sqlite"

    @staticmethod
    def __iter_storage_column_names(storage):
        """ Iter only those columns that existed in storage.
        """
        assert (isinstance(storage, RowCacheStorage))
        for col_name, col_type in zip(storage.iter_column_names(), storage.iter_column_types()):
            if col_name in storage.RowCache:
                yield col_name, col_type

    def __init_table(self, column_data):
        # Compose column name with the related SQLITE type.
        column_types = ",".join([" ".join([col_name, self.type_to_sqlite(col_type)])
                                 for col_name, col_type in column_data])
        # Create table if not exists.
        self.__cur.execute(f"CREATE TABLE IF NOT EXISTS {self.__table_name}({column_types})")
        # Table exists, however we may optionally remove the content from it.
        if self.__clear_table:
            self.__cur.execute(f"DELETE FROM {self.__table_name};")
        # Create index.
        index_name = f"i_{self.__table_name}_id"
        self.__cur.execute(f"DROP INDEX IF EXISTS {index_name};")
        self.__cur.execute("CREATE INDEX IF NOT EXISTS {index} ON {table}({columns})".format(
            index=index_name,
            table=self.__table_name,
            columns=", ".join(self.__index_column_names)
        ))
        self.__origin_column_names = [col_name for col_name, _ in column_data]

    @staticmethod
    def type_to_sqlite(col_type):
        """ This is a simple function that provides conversion from the
            base numpy types to SQLITE.
            NOTE: this method represent a quick implementation for supporting
            types, however it is far away from the generalized implementation.
        """
        if isinstance(col_type, str):
            if 'int' in col_type:
                return 'INTEGER'

        return "TEXT"

    def open_target(self, target):
        os.makedirs(dirname(target), exist_ok=True)
        self.__conn = sqlite3.connect(target)
        self.__cur = self.__conn.cursor()

    def commit_line(self, storage):
        assert (isinstance(storage, RowCacheStorage))

        column_data = list(self.__iter_storage_column_names(storage))

        if self.__need_init_table:
            self.__init_table(column_data)
            self.__need_init_table = False

        # Check whether the related row is already exist in SQLITE database.
        row_id = storage.RowCache[const.ID]
        top_row = self.__cur.execute(f"SELECT EXISTS(SELECT 1 FROM {self.__table_name} WHERE id='{row_id}');")
        is_exists = top_row.fetchone()[0]
        if is_exists == 1 and self.__skip_existed:
            return

        line_data = [storage.RowCache[col_name] for col_name, _ in column_data]
        parameters = ",".join(["?"] * len(line_data))

        assert (len(self.__origin_column_names) == len(line_data))

        self.__cur.execute(
            f"INSERT OR REPLACE INTO {self.__table_name} VALUES ({parameters})",
            tuple(line_data))

        self.__conn.commit()

    def close_target(self):
        self.__cur = None
        self.__origin_column_names = None
        self.__need_init_table = True
        self.__conn.close()

    def write_all(self, storage, target):
        pass
