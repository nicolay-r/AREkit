from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentrel.const import NEG_LABEL_STR, POS_LABEL_STR
from arekit.contrib.source.rusentrel.labels import PositiveLabel, NegativeLabel


class RuSentRelLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {NEG_LABEL_STR: self._negative_label_type(),
                POS_LABEL_STR: self._positive_label_type()}

        super(RuSentRelLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel()
