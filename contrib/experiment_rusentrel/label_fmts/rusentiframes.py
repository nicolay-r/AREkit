from arekit.contrib.experiment_rusentrel.labels import PositiveLabel, NegativeLabel
from arekit.contrib.source.rusentiframes.labels_fmt import \
    RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter


class ExperimentRuSentiFramesLabelsFormatter(RuSentiFramesLabelsFormatter):

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()


class ExperimentRuSentiFramesEffectLabelsFormatter(RuSentiFramesEffectLabelsFormatter):

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()
