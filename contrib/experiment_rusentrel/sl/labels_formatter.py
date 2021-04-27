from arekit.common.labels.base import NeutralLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentRelNeutralLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {u'neu': NeutralLabel()}
        super(RuSentRelNeutralLabelsFormatter, self).__init__(stol=stol)
