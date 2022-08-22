from arekit.common.labels.base import Label
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentiFramesLabelsFormatter(StringLabelsFormatter):

    def __init__(self, pos_label_type, neg_label_type):
        assert(issubclass(pos_label_type, Label))
        assert(issubclass(neg_label_type, Label))
        stol = {'neg': neg_label_type, 'pos': pos_label_type}
        super(RuSentiFramesLabelsFormatter, self).__init__(stol=stol)


class RuSentiFramesEffectLabelsFormatter(StringLabelsFormatter):
    """ Effect formatter utilizes '-' and '+' signs.
    """

    def __init__(self, pos_label_type, neg_label_type):
        assert(issubclass(pos_label_type, Label))
        assert(issubclass(neg_label_type, Label))
        stol = {'-': neg_label_type, '+': pos_label_type}
        super(RuSentiFramesEffectLabelsFormatter, self).__init__(stol=stol)
