import pandas as pd

from arekit.common.experiment import const
from arekit.common.experiment.input.views.base import BaseStorageView
from arekit.common.experiment.input.views.opinions import BaseOpinionStorageView
from arekit.common.experiment.row_ids.base import BaseIDProvider
from arekit.common.experiment.storages.base import BaseRowsStorage
from arekit.common.linked.opinions.wrapper import LinkedOpinionWrapper
from arekit.common.opinions.base import Opinion


class BaseOutputView(BaseStorageView):
    """ Results output represents a table, which stored in pandas dataframe.
        This dataframe assumes to provide the following columns:
            - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
            - news_id -- is a related news_id towards which the related output corresponds to.
            - labels -- uint labels (amount of columns depends on the scaler)
    """

    def __init__(self, ids_provider, storage):
        assert(isinstance(ids_provider, BaseIDProvider))
        assert(isinstance(storage, BaseRowsStorage))
        super(BaseOutputView, self).__init__(storage=storage)
        self._ids_provider = ids_provider

    # region private methods

    def __iter_linked_opinions_df(self, news_id):
        assert(isinstance(news_id, int))

        # TODO. Proceed refactoring
        news_df = self._storage.find_by_value(column_name=const.NEWS_ID,
                                              value=news_id)

        opinion_ids = [self._ids_provider.parse_opinion_in_opinion_id(opinion_id)
                       for opinion_id in news_df[const.ID]]

        for opinion_id in set(opinion_ids):
            opin_id_pattern = self._ids_provider.create_pattern(id_value=opinion_id,
                                                                p_type=BaseIDProvider.OPINION)
            linked_opins_df = news_df[news_df[const.ID].str.contains(opin_id_pattern)]
            yield linked_opins_df

    def __iter_linked_opinions(self, news_id, opinions_view):
        assert (isinstance(news_id, int))
        assert (isinstance(opinions_view, BaseOpinionStorageView))

        for linked_df in self.__iter_linked_opinions_df(news_id=news_id):
            assert (isinstance(linked_df, pd.DataFrame))

            opinions_iter = self._iter_by_opinions(linked_df=linked_df,
                                                   opinions_view=opinions_view)

            yield LinkedOpinionWrapper(linked_data=opinions_iter)

    # endregion

    # region protected methods

    def _get_column_header(self):
        raise NotImplementedError()

    def _iter_by_opinions(self, linked_df, opinions_view):
        raise NotImplementedError()

    def _compose_opinion_by_opinion_id(self, sample_id, opinions_view, calc_label_func):
        assert(isinstance(sample_id, str))
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        assert(callable(calc_label_func))

        opinion_id = self._ids_provider.convert_sample_id_to_opinion_id(sample_id=sample_id)
        source, target = opinions_view.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=calc_label_func())

    # endregion

    # region public methods

    def iter_news_ids(self):
        unique_news_ids = set(self._storage.iter_column_values(column_name=const.NEWS_ID))
        return unique_news_ids

    def iter_opinion_collections(self, opinions_view, keep_doc_id_func, to_collection_func):
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        assert(callable(keep_doc_id_func))
        assert(callable(to_collection_func))

        for news_id in self.iter_news_ids():

            if not keep_doc_id_func(news_id):
                continue

            linked_data_iter = self.__iter_linked_opinions(news_id=news_id,
                                                           opinions_view=opinions_view)

            yield news_id, to_collection_func(linked_data_iter)

    # endregion
