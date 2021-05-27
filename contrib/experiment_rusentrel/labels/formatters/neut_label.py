from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel


class RuSentRelNeutralLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {u'neu': ExperimentNeutralLabel()}
        super(RuSentRelNeutralLabelsFormatter, self).__init__(stol=stol)
