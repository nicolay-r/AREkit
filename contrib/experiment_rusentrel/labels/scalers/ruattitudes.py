from arekit.common.labels.base import NeutralLabel
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.source.ruattitudes.labels_scaler import RuAttitudesLabelScaler


class ExperimentRuAttitudesLabelScaler(RuAttitudesLabelScaler):

    @classmethod
    def _neutral_label_instance(cls):
        return NeutralLabel()

    @classmethod
    def _positive_label_instance(cls):
        return ExperimentPositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return ExperimentNegativeLabel()

