from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.storages.base import BaseRowsStorage


class BaseSampleStorageView(object):
    """
    Pandas-based input samples proovider
    """

    def __init__(self, storage, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        assert(isinstance(storage, BaseRowsStorage))
        self.__row_ids_provider = row_ids_provider
        self._storage = storage

    # TODO. #240 This is just a wrapper over storage.
    def iter_rows(self, handle_rows):
        assert(callable(handle_rows) or handle_rows is None)

        for row_index, row in self._storage:

            if handle_rows is None:
                yield row_index, row
            else:
                yield handle_rows(row)

    # TODO. #240 This is just a wrapper over storage.
    def extract_ids(self):
        return list(self._storage.iter_column_values(column_name=const.ID, dtype=str))

    # TODO. #240 This is just a wrapper over storage.
    def extract_doc_ids(self):
        return list(self._storage.iter_column_values(column_name=const.DOC_ID, dtype=int))

    def iter_rows_linked_by_text_opinions(self):
        undefined = -1

        linked = []

        current_doc_id = undefined
        current_opinion_id = undefined

        for row_index, sample_id in enumerate(self._storage.iter_column_values(const.ID)):
            sample_id = str(sample_id)

            doc_id = self._storage.get_cell(row_index=row_index, column_name=const.DOC_ID)
            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)

            if current_doc_id != undefined and current_opinion_id != undefined:
                if doc_id != current_doc_id or opinion_id != current_opinion_id:
                    yield linked
                    linked = []
            else:
                current_doc_id = doc_id
                current_opinion_id = opinion_id

            linked.append(self._storage.get_row(row_index))

        if len(linked) > 0:
            yield linked
