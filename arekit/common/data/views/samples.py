from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.storages.base import BaseRowsStorage


class BaseSampleStorageView(object):

    def __init__(self, storage, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        assert(isinstance(storage, BaseRowsStorage))
        # TODO. #269 make this provider as a part of the LinkedBasedStorageView (nested class from this)
        # TODO. #269. removed this parameter from here.
        self.__row_ids_provider = row_ids_provider
        self._storage = storage

    # TODO. #269 This is just a particular wrapper over storage.
    def iter_rows(self, row_handle_func):
        assert(callable(row_handle_func) or row_handle_func is None)

        for row_index, row in self._storage:

            if row_handle_func is None:
                yield row_index, row
            else:
                yield row_handle_func(row)

    # TODO. #269 This is just a particular wrapper over storage.
    def iter_rows_linked_by_text_opinions(self):
        undefined = -1

        linked = []

        current_opinion_id = undefined

        for row_index, sample_id in enumerate(self._storage.iter_column_values(const.ID)):
            sample_id = str(sample_id)

            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)

            if current_opinion_id != undefined:
                if opinion_id != current_opinion_id:
                    yield linked
                    linked = []
                    current_opinion_id = opinion_id
            else:
                current_opinion_id = opinion_id

            linked.append(self._storage.get_row(row_index))

        if len(linked) > 0:
            yield linked
