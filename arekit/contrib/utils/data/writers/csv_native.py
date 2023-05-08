import csv
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter


class NativeCsvWriter(BaseWriter):

    def __init__(self, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL):
        self.__target_f = None
        self.__writer = None
        self.__create_writer_func = lambda f: csv.writer(
            f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)

    def open_target(self, target):
        self.__target_f = open(target, "w")
        self.__writer = self.__create_writer_func(self.__target_f)
        pass

    def close_target(self):
        self.__target_f.close()

    def commit_line(self, storage):
        assert(isinstance(storage, RowCacheStorage))
        assert(self.__writer is not None)
        line_data = [storage.RowCache[col_name] for col_name in storage.iter_column_names()
                     if col_name in storage.RowCache]
        self.__writer.writerow(line_data)

    def write_all(self, storage, target):
        """ Writes all the `storage` rows
            into the `target` filepath, formatted as CSV.
        """
        assert(isinstance(storage, BaseRowsStorage))

        with open(target, "w") as f:
            writer = self.__create_writer_func(f)
            for _, row in storage:
                #content = [row[col_name] for col_name in storage.iter_column_names()]
                content = [v for v in row]
                writer.writerow(content)
