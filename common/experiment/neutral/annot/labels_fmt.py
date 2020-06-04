from arekit.common.labels.base import NegativeLabel, PositiveLabel, NeutralLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class ThreeScaleLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {u'neg': NegativeLabel(),
                u'pos': PositiveLabel(),
                u'neu': NeutralLabel()}

        super(ThreeScaleLabelsFormatter, self).__init__(stol=stol)
