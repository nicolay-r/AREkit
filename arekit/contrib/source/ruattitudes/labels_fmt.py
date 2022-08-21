from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuAttitudesLabelFormatter(StringLabelsFormatter):

    def __init__(self, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))
        stol = {}
        for uint_label in [0, 1, 2]:
            stol[str(uint_label)] = label_scaler.uint_to_label(uint_label)
        super(RuAttitudesLabelFormatter, self).__init__(stol=stol)
