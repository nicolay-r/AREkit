from arekit.common.experiment.scales.base import BaseLabelScaleExperiment
from arekit.common.labels.base import NeutralLabel, PositiveLabel, NegativeLabel


class TwoScaleExperiment(BaseLabelScaleExperiment):

    @staticmethod
    def supported_labels():
        return [NeutralLabel(), PositiveLabel(), NegativeLabel()]
