import pandas as pd

from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.base import BaseStorageView
from arekit.common.data.views.opinions import BaseOpinionStorageView
from arekit.common.linkage.opinions import OpinionsLinkage
from arekit.common.opinions.base import Opinion


class BaseOutputView(BaseStorageView):
    """ Results output represents a table, which stored in pandas dataframe.
        This dataframe assumes to provide the following columns:
            - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
            - doc_id -- is an id, towards which the output corresponds to.
            - labels -- uint labels (amount of columns depends on the scaler)
    """

    def __init__(self, ids_provider, storage):
        assert(isinstance(ids_provider, BaseIDProvider))
        assert(isinstance(storage, BaseRowsStorage))
        super(BaseOutputView, self).__init__(storage=storage)
        self._ids_provider = ids_provider

    # region private methods

    @staticmethod
    def __iter_opinion_linkages_df(doc_df, row_ids):
        for row_id in row_ids:
            df_linkage = doc_df[doc_df[const.ID].str.contains(row_id)]
            yield df_linkage

    def __iter_id_patterns(self, opinion_ids):
        for opinion_id in set(opinion_ids):
            yield self._ids_provider.create_pattern(id_value=opinion_id,
                                                    p_type=BaseIDProvider.OPINION)

    def __iter_doc_opinion_ids(self, doc_df):
        assert (isinstance(doc_df, pd.DataFrame))
        return [self._ids_provider.parse_opinion_in_opinion_id(row_id)
                for row_id in doc_df[const.ID]]

    def __iter_opinion_linkages(self, doc_df, opinions_view):
        assert (isinstance(doc_df, pd.DataFrame))
        assert (isinstance(opinions_view, BaseOpinionStorageView))

        doc_opin_ids = self.__iter_doc_opinion_ids(doc_df)
        doc_opin_id_patterns = self.__iter_id_patterns(doc_opin_ids)
        linkages_df = self.__iter_opinion_linkages_df(doc_df=doc_df,
                                                      row_ids=doc_opin_id_patterns)

        for df_linkage in linkages_df:
            assert (isinstance(df_linkage, pd.DataFrame))

            opinions_iter = self._iter_by_opinions(linked_df=df_linkage,
                                                   opinions_view=opinions_view)

            yield OpinionsLinkage(linked_data=opinions_iter)

    def __iter_doc_ids(self):
        return set(self._storage.iter_column_values(column_name=const.DOC_ID))

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

    def iter_opinion_collections(self, opinions_view, keep_doc_id_func, to_collection_func):
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        assert(callable(keep_doc_id_func))
        assert(callable(to_collection_func))

        # TODO. #237 __iter_doc_ids() should be utilized outside as a part of the pipeline.
        for doc_id in self.__iter_doc_ids():

            # TODO. #237 keep_doc_id_func(doc_id) should be utilized outside as a part of the pipeline.
            if not keep_doc_id_func(doc_id):
                continue

            # TODO. #237 find_by_value(doc_id) should be utilized outside + the latter should return Storage!
            doc_df = self._storage.find_by_value(column_name=const.DOC_ID,
                                                 value=doc_id)

            linkages_iter = self.__iter_opinion_linkages(doc_df=doc_df,
                                                         opinions_view=opinions_view)

            # TODO. #237 This to_collection_func(linkages_iter) should be outside and a part of the pipeline.
            collection = to_collection_func(linkages_iter)

            yield doc_id, collection

    # endregion
