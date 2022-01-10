from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentrel.const import \
    NEG_LABEL_STR, POS_LABEL_STR, \
    RUSENTREL_NEG_LABEL_TYPE, RUSENTREL_POS_LABEL_TYPE


class RuSentRelLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {NEG_LABEL_STR: self._negative_label_type(),
                POS_LABEL_STR: self._positive_label_type()}

        print(stol)

        super(RuSentRelLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_type(cls):
        return RUSENTREL_POS_LABEL_TYPE

    @classmethod
    def _negative_label_type(cls):
        return RUSENTREL_NEG_LABEL_TYPE
