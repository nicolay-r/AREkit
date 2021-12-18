from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNegativeLabel, ExperimentPositiveLabel
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter


class RuSentRelExperimentLabelsFormatter(RuSentRelLabelsFormatter):

    @classmethod
    def _negative_label_type(cls):
        return ExperimentNegativeLabel()

    @classmethod
    def _positive_label_type(cls):
        return ExperimentPositiveLabel()
