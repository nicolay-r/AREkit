from arekit.contrib.source.common.labels import NegativeLabel, PositiveLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentiFramesLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {u'neg': NegativeLabel(),
                u'pos': PositiveLabel()}

        super(RuSentiFramesLabelsFormatter, self).__init__(stol=stol)


class RuSentiFramesEffectLabelsFormatter(StringLabelsFormatter):
    """
    Effect formater utilizes '-' and '+' signs.
    """

    def __init__(self):

        stol = {u'-': NegativeLabel(),
                u'+': PositiveLabel()}

        super(RuSentiFramesEffectLabelsFormatter, self).__init__(stol=stol)
