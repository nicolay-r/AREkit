import pandas as pd

from arekit.bert.result.base import BertResults
from arekit.common.experiment.input.formatters.opinions.base import BertOpinionsFormatter
from arekit.common.experiment.input.providers.row_ids.base import BaseIDProvider
from arekit.bert.input.providers.row_ids.binary import BinaryIDProvider
from arekit.common.experiment.scales.base import BaseLabelScaler


class BertBinaryResults(BertResults):

    YES = u'yes'
    NO = u'no'

    def __init__(self, labels_scaler):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(BertBinaryResults, self).__init__(ids_formatter=BinaryIDProvider())
        self.__labels_scaler = labels_scaler

    @property
    def _IdsFormatter(self):
        formatter = super(BertBinaryResults, self)._IdsFormatter
        assert(isinstance(formatter, BinaryIDProvider))
        return formatter

    def _get_column_header(self):
        return [BertBinaryResults.NO, BertBinaryResults.YES]

    # region private methods

    def __calculate_label(self, df):
        """
        Calculate label by relying on a 'YES' column probability values
        paper: https://www.aclweb.org/anthology/N19-1035.pdf
        """
        ind_max = df[BertBinaryResults.YES].idxmax()
        sample_id = df.loc[ind_max][self.ID]
        uint_label = self._IdsFormatter.parse_label_in_sample_id(sample_id)
        return self.__labels_scaler.uint_to_label(value=uint_label)

    def _iter_by_opinions(self, linked_df, bert_opinions):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for opinion_ind in self.__iter_linked_opinion_indices(linked_df=linked_df):
            ind_pattern = self._IdsFormatter.create_pattern(id_value=opinion_ind,
                                                            p_type=BaseIDProvider.INDEX)
            opinion_df = linked_df[linked_df[self.ID].str.contains(ind_pattern)]

            yield self._compose_opinion_by_opinion_id(
                sample_id=opinion_df[self.ID].iloc[0],
                bert_opinions=bert_opinions,
                calc_label_func=lambda: self.__calculate_label(df=opinion_df))

    def __iter_linked_opinion_indices(self, linked_df):
        sample_ids = linked_df[self.ID].tolist()
        all_news = [self._IdsFormatter.parse_index_in_sample_id(sample_id)
                    for sample_id in sample_ids]
        for news_id in set(all_news):
            yield news_id

    # endregion
