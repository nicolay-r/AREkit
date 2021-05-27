from collections import OrderedDict

from arekit.common.labels.base import NoLabel, Label
from arekit.common.labels.scaler import BaseLabelScaler


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class TestThreeLabelScaler(BaseLabelScaler):

    def __init__(self):

        uint_labels = [(NoLabel(), 0),
                       (TestPositiveLabel(), 1),
                       (TestNegativeLabel(), 2)]

        int_labels = [(NoLabel(), 0),
                      (TestPositiveLabel(), 1),
                      (TestNegativeLabel(), -1)]

        super(TestThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                   int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
