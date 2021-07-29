from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel, \
    ExperimentNeutralLabel
from arekit.contrib.source.ruattitudes.labels_scaler import RuAttitudesLabelConverter


class ExperimentRuAttitudesLabelConverter(RuAttitudesLabelConverter):

    @classmethod
    def _neutral_label_instance(cls):
        return ExperimentNeutralLabel()

    @classmethod
    def _positive_label_instance(cls):
        return ExperimentPositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return ExperimentNegativeLabel()

