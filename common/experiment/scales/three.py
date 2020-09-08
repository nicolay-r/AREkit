from collections import OrderedDict

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import NeutralLabel, PositiveLabel, NegativeLabel


class ThreeLabelScaler(BaseLabelScaler):

    def __init__(self):

        uint_labels = [(NeutralLabel(), 0),
                       (PositiveLabel(), 1),
                       (NegativeLabel(), 2)]

        int_labels = [(NeutralLabel(), 0),
                      (PositiveLabel(), 1),
                      (NegativeLabel(), -1)]

        super(ThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                               int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.uint_to_label(-int_label)

    def __str__(self):
        return u"3"
