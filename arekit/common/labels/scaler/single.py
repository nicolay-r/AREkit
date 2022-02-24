from collections import OrderedDict

from arekit.common.labels.scaler.base import BaseLabelScaler


class SingleLabelScaler(BaseLabelScaler):

    def __init__(self, label, uint_label=0):
        d = OrderedDict([(label, uint_label)])
        super(SingleLabelScaler, self).__init__(uint_dict=d, int_dict=d)
