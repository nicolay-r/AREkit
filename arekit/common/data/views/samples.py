from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.storages.base import BaseRowsStorage


class LinkedSamplesStorageView(object):

    def __init__(self, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        self.__row_ids_provider = row_ids_provider

    def iter_from_storage(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        undefined = -1

        linked = []
        current_opinion_id = undefined
        for row_index, sample_id in enumerate(storage.iter_column_values(const.ID)):
            sample_id = str(sample_id)
            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)
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
