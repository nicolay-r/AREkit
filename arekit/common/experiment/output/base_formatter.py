import pandas as pd

from arekit.common.experiment import const
from arekit.common.experiment.input.providers.row_ids.base import BaseIDProvider
from arekit.common.experiment.input.readers.base_opinion import BaseInputOpinionReader
from arekit.common.experiment.output.base_provider import BaseOutputProvider
from arekit.common.linked.opinions.wrapper import LinkedOpinionWrapper
from arekit.common.opinions.base import Opinion


class BaseOutputFormatter(object):
    """ Results output represents a table, which stored in pandas dataframe.
        This dataframe assumes to provide the following columns:
            - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
            - news_id -- is a related news_id towards which the related output corresponds to.
            - labels -- uint labels (amount of columns depends on the scaler)
    """

    def __init__(self, ids_formatter, output_provider):
        assert(isinstance(ids_formatter, BaseIDProvider))
        assert(isinstance(output_provider, BaseOutputProvider))
        self.__ids_formatter = ids_formatter
        self.__provider = output_provider

    # region properties

    @property
    def _IdsFormatter(self):
        return self.__ids_formatter

    @property
    def _df(self):
        return self.__provider.DataFrame

    # endregion

    # region private methods

    def __iter_linked_opinions_df(self, news_id):
        assert(isinstance(news_id, int))

        news_df = self._df[self._df[const.NEWS_ID] == news_id]
        opinion_ids = [self.__ids_formatter.parse_opinion_in_opinion_id(opinion_id)
                       for opinion_id in news_df[const.ID]]

        for opinion_id in set(opinion_ids):
            opin_id_pattern = self.__ids_formatter.create_pattern(id_value=opinion_id,
                                                                  p_type=BaseIDProvider.OPINION)
            linked_opins_df = news_df[news_df[const.ID].str.contains(opin_id_pattern)]
            yield linked_opins_df

    # endregion

    # region protected methods

    def _get_column_header(self):
        raise NotImplementedError()

    def _iter_by_opinions(self, linked_df, opinions_reader):
        raise NotImplementedError()

    def _compose_opinion_by_opinion_id(self, sample_id, opinions_reader, calc_label_func):
        assert(isinstance(sample_id, str))
        assert(isinstance(opinions_reader, BaseInputOpinionReader))
        assert(callable(calc_label_func))

        opinion_id = self.__ids_formatter.convert_sample_id_to_opinion_id(sample_id=sample_id)
        source, target = opinions_reader.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=calc_label_func())

    # endregion

    # region public methods

    def iter_news_ids(self):
        assert (const.NEWS_ID in self._df.columns)
        return set(self._df[const.NEWS_ID])

    def iter_linked_opinions(self, news_id, opinions_reader):
        assert (isinstance(news_id, int))
        assert (isinstance(opinions_reader, BaseInputOpinionReader))

        for linked_df in self.__iter_linked_opinions_df(news_id=news_id):
            assert (isinstance(linked_df, pd.DataFrame))

            opinions_iter = self._iter_by_opinions(linked_df=linked_df,
                                                   opinions_reader=opinions_reader)

            yield LinkedOpinionWrapper(linked_data=opinions_iter)

    def load(self, source):
        self.__provider.load(source)

    # endregion

    def __len__(self):
        return len(self._df.index)
