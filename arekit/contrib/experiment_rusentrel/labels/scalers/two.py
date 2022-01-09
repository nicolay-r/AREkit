from collections import OrderedDict

from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel


class TwoLabelScaler(SentimentLabelScaler):

    def __init__(self):

        uint_labels = [(ExperimentPositiveLabel(), 0),
                       (ExperimentNegativeLabel(), 1)]

        super(TwoLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                             int_dict=OrderedDict(uint_labels))

    def invert_label(self, label):
        uint_label = self.label_to_uint(label)
        return self.uint_to_label(1 - uint_label)

