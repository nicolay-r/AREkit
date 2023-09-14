import sqlite3

from arekit.common.data import const
from arekit.contrib.utils.data.writers.base import BaseWriter


class SQliteWriter(BaseWriter):

    def __init__(self, table_name="contents"):
        self.__table_name = table_name
        self.__conn = None
        self.__cur = None
        self.__need_init_table = True
        self.__column_names = None

    @staticmethod
    def __iter_storage_column_names(storage):
        """ Iter only those columns that existed in storage.
        """
        for col_name in storage.iter_column_names():
            if col_name in storage.RowCache:
                yield col_name

    def open_target(self, target):
        self.__conn = sqlite3.connect(target + ".sqlite")
        self.__cur = self.__conn.cursor()

    def commit_line(self, storage):

        column_names = list(self.__iter_storage_column_names(storage))

        if self.__need_init_table:
            column_types = ",".join(["{} {}".format(item[0], item[1])
                                     for item in zip(column_names, ["TEXT"] * len(column_names))])

            self.__cur.execute(f"CREATE TABLE IF NOT EXISTS {self.__table_name}({column_types})")
            self.__cur.execute(f"CREATE INDEX IF NOT EXISTS i_id ON {self.__table_name}({const.ID})")

            self.__column_names = column_names
            self.__need_init_table = False

        row_id = storage.RowCache[const.ID]
        top_row = self.__cur.execute(f"SELECT EXISTS(SELECT 1 FROM {self.__table_name} WHERE id='{row_id}');")
        is_exists = top_row.fetchone()[0]
        if is_exists == 1:
            return

        line_data = [storage.RowCache[col_name] for col_name in column_names]
        parameters = ",".join(["?"] * len(line_data))

        assert(len(self.__column_names) == len(line_data))

        self.__cur.execute(
            f"INSERT INTO {self.__table_name} VALUES ({parameters})",
            tuple(line_data))

        self.__conn.commit()

    def close_target(self):
        self.__cur = None
        self.__column_names = None
        self.__need_init_table = True
        self.__conn.close()

    def write_all(self, storage, target):
        pass
