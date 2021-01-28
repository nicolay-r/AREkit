from arekit.common.experiment.scales.base import BaseLabelScaler


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

    # TODO. Use uint_label
    def calculate_output_label(self, expected_label, etalon_label):
        raise NotImplementedError()

