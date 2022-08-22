from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentiframes.labels import PositiveLabel, NegativeLabel


class RuSentiFramesLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {'neg': self._negative_label_type(),
                'pos': self._positive_label_type()}

        super(RuSentiFramesLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel


class RuSentiFramesEffectLabelsFormatter(StringLabelsFormatter):
    """
    Effect formatter utilizes '-' and '+' signs.
    """

    def __init__(self):

        stol = {'-': self._negative_label_type(),
                '+': self._positive_label_type()}

        super(RuSentiFramesEffectLabelsFormatter, self).__init__(stol=stol)

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel
