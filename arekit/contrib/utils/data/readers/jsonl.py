from arekit.contrib.utils.data.readers.base import BaseReader
from arekit.contrib.utils.data.storages.jsonl_based import JsonlBasedRowsStorage


class JsonlReader(BaseReader):

    def read(self, target):
        rows = []
        with open(target, "r") as f:
            for line in f.readlines():
                rows.append(line)
        return JsonlBasedRowsStorage(rows)
