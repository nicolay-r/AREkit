from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage


# TODO. This is a particular type of view, and expected to be off the core.
class LinkedSamplesStorageView(object):

    def iter_from_storage(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        undefined = -1

        linked = []
        current_opinion_id = undefined
        for row_index, opinion_id in enumerate(storage.iter_column_values(const.OPINION_ID)):
            if current_opinion_id != undefined:
                if opinion_id != current_opinion_id:
                    yield linked
                    linked = []
                    current_opinion_id = opinion_id
            else:
                current_opinion_id = opinion_id

            linked.append(storage.get_row(row_index))

        if len(linked) > 0:
            yield linked
