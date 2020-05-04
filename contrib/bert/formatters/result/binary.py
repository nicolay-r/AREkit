import pandas as pd

from arekit.common.experiment.scales.base import BaseLabelScaler
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

    def __calculate_label(self, opinion_df):
        """
        Calculate label by relying on a 'YES' column probability values
        paper: https://www.aclweb.org/anthology/N19-1035.pdf
        """
        ind_max = opinion_df[BertBinaryResults.YES].idxmax()
        related_id = opinion_df[self.ID].iloc[ind_max]
        uint_label = self._IdsFormatter.parse_label_in_sample_id(related_id)
        return self.__labels_scaler.uint_to_label(value=uint_label)

    def iter_wrapped_linked_text_opinions(self, news_id, bert_opinions):
        assert(isinstance(news_id, int))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for linked_df in self._iter_linked_opinions_df(news_id=news_id):
            assert(isinstance(linked_df, pd.DataFrame))
            for index in self.__iter_linked_opinion_indices(linked_df=linked_df):

                # Extract related part in order to calculate label
                # and define text opinion ends

                pattern = self._IdsFormatter.create_index_id_pattern(index)
                opinion_df = linked_df[linked_df[self.ID].str.contains(pattern)]
                self.__calculate_label(opinion_df=opinion_df)
                _, source, target = bert_opinions.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

                return Opinion(source_value=source,
                               target_value=target,
                               sentiment=self._calculate_label(row))

    def __to_wrapped_linked_opinions(self):

    def __iter_linked_opinion_indices(self, linked_df):
        ids = linked_df[self.ID].tolist()
        all_news = [self.__ids_formatter.parse_index_in_sample_id(row_id) for row_id in ids]
        for news_id in set(all_news):
            yield news_id
