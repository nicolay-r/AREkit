from arekit.common.experiment import const
from arekit.common.experiment.input.providers.row_ids.base import BaseIDProvider
from arekit.common.experiment.input.readers.base import BaseInputReader


class BaseInputSampleReader(BaseInputReader):
    """
    Pandas-based input samples proovider
    """

    def __init__(self, df, row_ids_provider):
        assert(isinstance(row_ids_provider, BaseIDProvider))
        super(BaseInputSampleReader, self).__init__(df)

        self.__row_ids_provider = row_ids_provider

    def extract_ids(self):
        return self._df[const.ID].astype(str).tolist()

    def iter_news_ids(self):
        return self._df[const.NEWS_ID].astype(int).tolist()

    def iter_rows_linked_by_text_opinions(self):
        undefined = -1

        linked = []

        current_news_id = undefined
        current_opinion_id = undefined

        for row_index, sample_id in enumerate(self._df[const.ID]):
            sample_id = str(sample_id)

            news_id = self._df.iloc[row_index][const.NEWS_ID]
            opinion_id = self.__row_ids_provider.parse_opinion_in_sample_id(sample_id)

            if current_news_id != undefined and current_opinion_id != undefined:
                if news_id != current_news_id or opinion_id != current_opinion_id:
                    yield linked
                    linked = []
            else:
                current_news_id = news_id
                current_opinion_id = opinion_id

            linked.append(self._df.iloc[row_index])

        if len(linked) > 0:
            yield linked
