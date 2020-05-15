import numpy as np
import pandas as pd

from arekit.common.experiment.scales.base import BaseLabelScaler

from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.result.base import BertResults


class BertMultipleResults(BertResults):

    def __init__(self, labels_scaler):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(BertMultipleResults, self).__init__(ids_formatter=MultipleIDFormatter())
        self.__labels_scaler = labels_scaler

    # region protected methods

    def _get_column_header(self):
        # TODO. Provide formatter str (or expand existed labels formatter).
        return [label.to_str() for label in self.__labels_scaler.ordered_suppoted_labels()]

    def __calculate_label(self, row):
        """
        Using a single row (probabilities by each class)
        """
        labels_prob = [row[label] for label in self._get_column_header()]
        return self.__labels_scaler.uint_to_label(value=np.argmax(labels_prob))

    def _iter_by_opinions(self, linked_df, bert_opinions):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for index, series in linked_df.iterrows():
            yield self._compose_opinion_by_opinion_id(
                sample_id=series[self.ID],
                bert_opinions=bert_opinions,
                calc_label_func=lambda: self.__calculate_label(series))

    # endregion

