from arekit.common.experiment.input.storages.base import BaseRowsStorage


class BaseWriter(object):

    def __init__(self):
        self._storage = None

    def set_storage(self, value):
        assert(isinstance(value, BaseRowsStorage))
        self._storage = value

    def save(self, target):
        raise NotImplementedError()
