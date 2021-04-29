from arekit.contrib.source.common.labels import NegativeLabel, PositiveLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentiFramesLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {u'neg': self._negative_label_instance(),
                u'pos': self._positive_label_instance()}

        super(RuSentiFramesLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()


class RuSentiFramesEffectLabelsFormatter(StringLabelsFormatter):
    """
    Effect formatter utilizes '-' and '+' signs.
    """

    def __init__(self):

        stol = {u'-': self._negative_label_instance(),
                u'+': self._positive_label_instance()}

        super(RuSentiFramesEffectLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()
