from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.source.rusentiframes.labels_fmt import \
    RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter


class ExperimentRuSentiFramesLabelsFormatter(RuSentiFramesLabelsFormatter):

    @classmethod
    def _positive_label_type(cls):
        return ExperimentPositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return ExperimentNegativeLabel


class ExperimentRuSentiFramesEffectLabelsFormatter(RuSentiFramesEffectLabelsFormatter):

    @classmethod
    def _positive_label_type(cls):
        return ExperimentPositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return ExperimentNegativeLabel
