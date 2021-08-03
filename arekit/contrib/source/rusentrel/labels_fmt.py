from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentrel.const import \
    NEG_LABEL_STR, POS_LABEL_STR, \
    RUSENTREL_NEG_LABEL, RUSENTREL_POS_LABEL


class RuSentRelLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        pos_label = self._positive_label_instance()
        neg_label = self._negative_label_instance()

        stol = {NEG_LABEL_STR: neg_label,
                POS_LABEL_STR: pos_label}

        super(RuSentRelLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_instance(cls):
        return RUSENTREL_POS_LABEL

    @classmethod
    def  _negative_label_instance(cls):
        return RUSENTREL_NEG_LABEL
