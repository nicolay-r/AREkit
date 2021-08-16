from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel


class ExperimentNeutralLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {'neu': ExperimentNeutralLabel()}
        super(ExperimentNeutralLabelsFormatter, self).__init__(stol=stol)
