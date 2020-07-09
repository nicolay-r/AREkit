import numpy as np
import pandas as pd

from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.scales.base import BaseLabelScaler


class MulticlassOutput(BaseOutput):

    def __init__(self, labels_scaler):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(MulticlassOutput, self).__init__(ids_formatter=MultipleIDProvider())
        self.__labels_scaler = labels_scaler

    # region protected methods

    def _get_column_header(self):
        return [str(self.__labels_scaler.label_to_uint(label))
                for label in self.__labels_scaler.ordered_suppoted_labels()]

    def __calculate_label(self, row):
        """
        Using a single row (probabilities by each class)
        """
        labels_prob = [row[label] for label in self._get_column_header()]
        return self.__labels_scaler.uint_to_label(value=np.argmax(labels_prob))

    # TODO. Replace with opinions reader.
    def _iter_by_opinions(self, linked_df, opinions_formatter):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(opinions_formatter, BaseOpinionsFormatter))

        for index, series in linked_df.iterrows():
            yield self._compose_opinion_by_opinion_id(
                sample_id=series[self.ID],
                # TODO. Replace with opinions reader.
                opinions_formatter=opinions_formatter,
                calc_label_func=lambda: self.__calculate_label(series))

    # endregion

