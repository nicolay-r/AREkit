from arekit.contrib.source.common.labels import NegativeLabel, PositiveLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentRelLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {u'neg': NegativeLabel(),
                u'pos': PositiveLabel()}

        super(RuSentRelLabelsFormatter, self).__init__(stol=stol)
