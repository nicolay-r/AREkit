from collections import OrderedDict

from arekit.common.labels.base import Label, NoLabel
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler


class TestNeutralLabel(NoLabel):
    pass


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class TestThreeLabelScaler(SentimentLabelScaler):

    def __init__(self):

        uint_labels = [(TestNeutralLabel(), 0),
                       (TestPositiveLabel(), 1),
                       (TestNegativeLabel(), 2)]

        int_labels = [(TestNeutralLabel(), 0),
                      (TestPositiveLabel(), 1),
                      (TestNegativeLabel(), -1)]

        super(TestThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                   int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
