from arekit.common.labels.scaler.base import BaseLabelScaler


class LabelProvider(object):

    def __init__(self, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))
        self.__label_scaler = label_scaler

    @property
    def LabelScaler(self):
        return self.__label_scaler

    @property
    def SupportedLabels(self):
        return self.__label_scaler.ordered_suppoted_labels()

    @property
    def OutputLabelsUint(self):
        raise NotImplementedError()

    def calculate_output_uint_label(self, expected_uint_label, etalon_uint_label):
        raise NotImplementedError()

