from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage


class BaseOpinionStorageView(object):

    def __init__(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        self._storage = storage

    def row_by_id(self, opinion_id):
        assert(isinstance(opinion_id, str))
        return self._storage.find_first_by_value(column_name=const.ID,
                                                 value=opinion_id)
