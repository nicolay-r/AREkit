from collections import OrderedDict

from arekit.common.labels.base import Label
from arekit.common.labels.scaler.base import BaseLabelScaler


class NoLabel(Label):
    pass


class TestNeutralLabel(NoLabel):
    pass


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class SentimentLabelScaler(BaseLabelScaler):

    def __init__(self):
        int_to_label = OrderedDict([(TestNeutralLabel(), 0), (TestPositiveLabel(), 1), (TestNegativeLabel(), -1)])
        uint_to_label = OrderedDict([(TestNeutralLabel(), 0), (TestPositiveLabel(), 1), (TestNegativeLabel(), 2)])
        super(SentimentLabelScaler, self).__init__(int_to_label, uint_to_label)


class TestThreeLabelScaler(SentimentLabelScaler):

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
