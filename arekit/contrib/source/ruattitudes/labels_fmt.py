from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuAttitudesLabelFormatter(StringLabelsFormatter):

    def __init__(self, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))
        stol = {}
        for int_label in [-1, 0, 1]:
            stol[str(int_label)] = type(label_scaler.int_to_label(int_label))
        super(RuAttitudesLabelFormatter, self).__init__(stol=stol)
