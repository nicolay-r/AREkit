from arekit.common.labels.base import Label
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentrel.const import NEG_LABEL_STR, POS_LABEL_STR


class RuSentRelLabelsFormatter(StringLabelsFormatter):

    def __init__(self, pos_label_type, neg_label_type):
        assert(issubclass(pos_label_type, Label))
        assert(issubclass(neg_label_type, Label))
        stol = {NEG_LABEL_STR: neg_label_type, POS_LABEL_STR: pos_label_type}
        super(RuSentRelLabelsFormatter, self).__init__(stol=stol)
