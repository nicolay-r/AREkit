from collections import OrderedDict

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import NeutralLabel, PositiveLabel, NegativeLabel


class ThreeLabelScaler(BaseLabelScaler):

    def __init__(self):
        labels = [(NeutralLabel(), 0),
                  (PositiveLabel(), 1),
                  (NegativeLabel(), 2)]

        super(ThreeLabelScaler, self).__init__(to_uint=OrderedDict(labels))
