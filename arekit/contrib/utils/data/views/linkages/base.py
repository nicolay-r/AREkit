import pandas as pd

from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.linkage.opinions import OpinionsLinkage
from arekit.contrib.utils.data.views.linkages import utils
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView


class BaseOpinionLinkagesView(object):
    """ Base view onto source in terms of opinion linkages.
    """

    def __init__(self, ids_provider, storage):
        assert(isinstance(ids_provider, BaseIDProvider))
        assert(isinstance(storage, BaseRowsStorage))
        self._ids_provider = ids_provider
        self._storage = storage

    # region private methods

    def __iter_opinions_by_linkages(self, linkages_df, opinions_view):
        for df_linkage in linkages_df:
            assert (isinstance(df_linkage, pd.DataFrame))
            yield self._iter_by_opinions(linked_df=df_linkage, opinions_view=opinions_view)

    # endregion

    # region protected methods

    def _iter_by_opinions(self, linked_df, opinions_view):
        raise NotImplementedError()

    # endregion

    # region public methods

    def iter_opinion_linkages(self, doc_id, opinions_view):
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        doc_df = self._storage.find_by_value(column_name=const.OPINION_ID, value=doc_id)
        doc_opin_ids = [opinion_id for opinion_id in doc_df[const.OPINION_ID]]

        doc_opin_id_patterns = map(
            lambda opinion_id: self._ids_provider.create_pattern(id_value=opinion_id, p_type=BaseIDProvider.OPINION),
            doc_opin_ids)

        linkages_df = map(
            lambda opin_id: utils.filter_by_id(doc_df=doc_df, column=const.ID, value=opin_id),
            doc_opin_id_patterns)

        opinions_iter = self.__iter_opinions_by_linkages(linkages_df, opinions_view=opinions_view)

        return map(lambda opinions: OpinionsLinkage(opinions), opinions_iter)

    # endregion
