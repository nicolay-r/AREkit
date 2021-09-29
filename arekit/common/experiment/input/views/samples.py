from arekit.common.experiment import const
from arekit.common.experiment.input.views.base import BaseStorageView
from arekit.common.experiment.row_ids.base import BaseIDProvider


class BaseSampleStorageView(BaseStorageView):
    """
    Pandas-based input samples proovider
    """

    def __init__(self, storage, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        super(BaseSampleStorageView, self).__init__(storage)
        self.__row_ids_provider = row_ids_provider

    def extract_ids(self):
        return list(self._storage.iter_column_values(column_name=const.ID, dtype=str))

    def extract_news_ids(self):
        return list(self._storage.iter_column_values(column_name=const.NEWS_ID, dtype=int))

    def iter_rows_linked_by_text_opinions(self):
        undefined = -1

        linked = []

        current_news_id = undefined
        current_opinion_id = undefined

        for row_index, sample_id in enumerate(self._storage.iter_column_values(const.ID)):
            sample_id = str(sample_id)

            news_id = self._storage.get_cell(row_index=row_index, column_name=const.NEWS_ID)
            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)

            if current_news_id != undefined and current_opinion_id != undefined:
                if news_id != current_news_id or opinion_id != current_opinion_id:
                    yield linked
                    linked = []
            else:
                current_news_id = news_id
                current_opinion_id = opinion_id

            linked.append(self._storage.get_row(row_index))

        if len(linked) > 0:
            yield linked

    def calculate_news_id_by_sample_id_dict(self):
        """
        Iter sample_ids with the related labels (if the latter presented in dataframe)
        """
        news_id_by_sample_id = {}

        for row_index, row in self._storage:

            sample_id = row[const.ID]

            if sample_id in news_id_by_sample_id:
                continue

            news_id_by_sample_id[sample_id] = row[const.NEWS_ID]

        return news_id_by_sample_id
