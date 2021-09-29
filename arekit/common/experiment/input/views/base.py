from arekit.common.experiment.input.storages.base import BaseRowsStorage


class BaseStorageView(object):

    def __init__(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        self._storage = storage

    def iter_handled_rows(self, handle_rows):
        assert(callable(handle_rows))
        for row_index, row in self._storage:
            yield handle_rows(row)
