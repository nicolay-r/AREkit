from arekit.common.data.input.readers.base import BaseReader
from arekit.common.data.storages.base import BaseRowsStorage


class TsvReader(BaseReader):
    def read(self, target):
        return BaseRowsStorage.from_tsv(filepath=target)
