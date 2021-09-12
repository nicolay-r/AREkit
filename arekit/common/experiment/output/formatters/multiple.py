import numpy as np
import pandas as pd

from arekit.common.experiment import const
from arekit.common.experiment.output.formatters.base import BaseOutputFormatter
from arekit.common.experiment.input.readers.base_opinion import BaseInputOpinionReader
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.labels.scaler import BaseLabelScaler


class MulticlassOutputFormatter(BaseOutputFormatter):

    def __init__(self, labels_scaler, output_provider):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(MulticlassOutputFormatter, self).__init__(ids_formatter=MultipleIDProvider(),
                                                        output_provider=output_provider)
        self.__labels_scaler = labels_scaler

    # region private methods

    def __calculate_label(self, row):
        """
        Using a single row (probabilities by each class)
        """
        labels_prob = [row[label] for label in self._get_column_header()]
        return self.__labels_scaler.uint_to_label(value=np.argmax(labels_prob))

    # endregion

    # region protected methods

    def _get_column_header(self):
        return [str(self.__labels_scaler.label_to_uint(label))
                for label in self.__labels_scaler.ordered_suppoted_labels()]

    def _iter_by_opinions(self, linked_df, opinions_reader):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(opinions_reader, BaseInputOpinionReader))

        for index, series in linked_df.iterrows():
            yield self._compose_opinion_by_opinion_id(
                sample_id=series[const.ID],
                opinions_reader=opinions_reader,
                calc_label_func=lambda: self.__calculate_label(series))

    # endregion
