from collections import OrderedDict

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.base import PositiveLabel, NegativeLabel


class TwoLabelScaler(BaseLabelScaler):

    def __init__(self):
        labels = [(PositiveLabel(), 0),
                  (NegativeLabel(), 1)]

        super(TwoLabelScaler, self).__init__(to_uint=OrderedDict(labels))

