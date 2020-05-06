import pandas as pd

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.opinions.base import Opinion
from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.result.base import BertResults
from arekit.contrib.bert.formatters.row_ids.binary import BinaryIDFormatter


class BertBinaryResults(BertResults):

    YES = u'yes'
    NO = u'no'

    def __init__(self, labels_scaler):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(BertBinaryResults, self).__init__(ids_formatter=BinaryIDFormatter())
        self.__labels_scaler = labels_scaler

    @property
    def _IdsFormatter(self):
        formatter = super(BertBinaryResults, self)._IdsFormatter()
        assert(isinstance(formatter, BinaryIDFormatter))
        return formatter

    def _get_column_header(self):
        return [BertBinaryResults.NO, BertBinaryResults.YES]

    # region private methods

    def __calculate_label(self, opinion_df):
        """
        Calculate label by relying on a 'YES' column probability values
        paper: https://www.aclweb.org/anthology/N19-1035.pdf
        """
        ind_max = opinion_df[BertBinaryResults.YES].idxmax()
        related_id = opinion_df[self.ID].iloc[ind_max]
        uint_label = self._IdsFormatter.parse_label_in_sample_id(related_id)
        return self.__labels_scaler.uint_to_label(value=uint_label)

    def _iter_by_opinions(self, linked_df, bert_opinions):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(bert_opinions, bert_opinions))

        for opinion_ind in self.__iter_linked_opinion_indices(linked_df=linked_df):
            ind_pattern = self._IdsFormatter.create_index_id_pattern(opinion_ind)
            opinion_df = linked_df[linked_df[self.ID].str.contains(ind_pattern)]

            yield self._compose_opinion_by_opinion_id(
                sample_id=opinion_df[self.ID].iloc[0],
                bert_opinions=bert_opinions,
                calc_label_func=lambda: self.__calculate_label(opinion_df=opinion_df))

    def __iter_linked_opinion_indices(self, linked_df):
        ids = linked_df[self.ID].tolist()
        all_news = [self.__ids_formatter.parse_index_in_sample_id(row_id) for row_id in ids]
        for news_id in set(all_news):
            yield news_id

    # endregion
