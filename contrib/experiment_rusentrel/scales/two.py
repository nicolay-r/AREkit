from collections import OrderedDict

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.contrib.experiment_rusentrel.labels import NegativeLabel, PositiveLabel


class TwoLabelScaler(BaseLabelScaler):

    def __init__(self):

        uint_labels = [(PositiveLabel(), 0),
                       (NegativeLabel(), 1)]

        super(TwoLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                             int_dict=OrderedDict(uint_labels))

    def invert_label(self, label):
        uint_label = self.label_to_uint(label)
        return self.uint_to_label(1 - uint_label)

