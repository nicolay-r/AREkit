from arekit.contrib.experiment_rusentrel.labels import PositiveLabel, NegativeLabel
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter


class RuSentRelExperimentLabelsFormatter(RuSentRelLabelsFormatter):

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()
