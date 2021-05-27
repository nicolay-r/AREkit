from collections import OrderedDict

from arekit.common.labels.base import NeutralLabel
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNegativeLabel, ExperimentPositiveLabel


class ThreeLabelScaler(BaseLabelScaler):

    def __init__(self):

        uint_labels = [(NeutralLabel(), 0),
                       (ExperimentPositiveLabel(), 1),
                       (ExperimentNegativeLabel(), 2)]

        int_labels = [(NeutralLabel(), 0),
                      (ExperimentPositiveLabel(), 1),
                      (ExperimentNegativeLabel(), -1)]

        super(ThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                               int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)