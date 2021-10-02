import pandas as pd

from arekit.common.experiment.input.views.opinions import BaseOpinionStorageView
from arekit.common.experiment.output.views.base import BaseOutputView
from arekit.common.experiment.row_ids.base import BaseIDProvider
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.experiment import const
from arekit.contrib.bert.input.providers.row_ids_binary import BinaryIDProvider


class BertBinaryOutputView(BaseOutputView):

    YES = 'yes'
    NO = 'no'

    def __init__(self, labels_scaler, storage):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(BertBinaryOutputView, self).__init__(ids_provider=BinaryIDProvider(),
                                                   storage=storage)
        self.__labels_scaler = labels_scaler

    # region private methods

    def __calculate_label(self, df):
        """
        Calculate label by relying on a 'YES' column probability values
        paper: https://www.aclweb.org/anthology/N19-1035.pdf
        """
        ind_max = df[BertBinaryOutputView.YES].idxmax()
        sample_id = df.loc[ind_max][const.ID]
        uint_label = self._ids_provider.parse_label_in_sample_id(sample_id)
        return self.__labels_scaler.uint_to_label(value=uint_label)

    def __iter_linked_opinion_indices(self, linked_df):
        sample_ids = linked_df[const.ID].tolist()
        all_news = [self._ids_provider.parse_index_in_sample_id(sample_id)
                    for sample_id in sample_ids]
        for news_id in set(all_news):
            yield news_id

    # endregion

    # region protected methods

    def _get_column_header(self):
        return [BertBinaryOutputView.NO, BertBinaryOutputView.YES]

    def _iter_by_opinions(self, linked_df, opinions_view):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(opinions_view, BaseOpinionStorageView))

        for opinion_ind in self.__iter_linked_opinion_indices(linked_df=linked_df):
            ind_pattern = self._ids_provider.create_pattern(id_value=opinion_ind,
                                                            p_type=BaseIDProvider.INDEX)
            opinion_df = linked_df[linked_df[const.ID].str.contains(ind_pattern)]

            yield self._compose_opinion_by_opinion_id(
                sample_id=opinion_df[const.ID].iloc[0],
                opinions_view=opinions_view,
                calc_label_func=lambda: self.__calculate_label(df=opinion_df))

    # endregion
