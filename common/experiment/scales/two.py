from arekit.common.experiment.scales.base import BaseLabelScaleExperiment
from arekit.common.labels.base import PositiveLabel, NegativeLabel


class ThreeScaleExperiment(BaseLabelScaleExperiment):

    @staticmethod
    def supported_labels():
        return [PositiveLabel(), NegativeLabel()]
