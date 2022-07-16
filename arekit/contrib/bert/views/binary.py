import pandas as pd

from arekit.common.data import const
from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.bert.input.providers.row_ids_binary import BinaryIDProvider
from arekit.contrib.utils.data.views.linkages import utils
from arekit.contrib.utils.data.views.linkages.base import BaseOpinionLinkagesView
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView


class BertBinaryOpinionLinkagesView(BaseOpinionLinkagesView):

    YES = 'yes'
    NO = 'no'

    def __init__(self, labels_scaler, storage):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(BertBinaryOpinionLinkagesView, self).__init__(ids_provider=BinaryIDProvider(),
                                                            storage=storage)
        self.__labels_scaler = labels_scaler

    # region private methods

    def __calculate_label(self, df):
        """
        Calculate label by relying on a 'YES' column probability values
        paper: https://www.aclweb.org/anthology/N19-1035.pdf
        """
        ind_max = df[BertBinaryOpinionLinkagesView.YES].idxmax()
        sample_id = df.loc[ind_max][const.ID]
        uint_label = self._ids_provider.parse_label_in_sample_id(sample_id)
        return self.__labels_scaler.uint_to_label(value=uint_label)

    def __iter_linked_opinion_indices(self, linked_df):
        sample_ids = linked_df[const.ID].tolist()
        all_docs = [self._ids_provider.parse_index_in_sample_id(sample_id)
                    for sample_id in sample_ids]
        for doc_id in set(all_docs):
            yield doc_id

    # endregion

    # region protected methods

    def _iter_by_opinions(self, linked_df, opinions_view):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(opinions_view, BaseOpinionStorageView))

        opinion_ids = self.__iter_linked_opinion_indices(linked_df=linked_df)

        id_patterns_iter = map(
            lambda opinion_id: self._ids_provider.create_pattern(id_value=opinion_id,
                                                                 p_type=BaseIDProvider.INDEX),
            opinion_ids)

        linkages_dfs = map(
            lambda id_pattern: utils.filter_by_id(doc_df=linked_df,
                                                  column=const.ID,
                                                  value=id_pattern),
            id_patterns_iter)

        for opinion_linkage_df in linkages_dfs:
            yield utils.compose_opinion_by_opinion_id(
                ids_provider=self._ids_provider,
                sample_id=opinion_linkage_df[const.ID].iloc[0],
                opinions_view=opinions_view,
                calc_label_func=lambda: self.__calculate_label(df=opinion_linkage_df))

    # endregion
