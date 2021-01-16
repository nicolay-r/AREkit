from arekit.common.experiment.input.providers.label.base import LabelProvider
from arekit.common.labels.base import Label


class MultipleLabelProvider(LabelProvider):

    def __init__(self, label_scaler):
        super(MultipleLabelProvider, self).__init__(label_scaler=label_scaler)

    # TODO. Use uint_label in param
    def calculate_output_label(self, expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        # TODO. Use uint_label
        return self.LabelScaler.label_to_uint(label=expected_label)

    @property
    def OutputLabelsUint(self):
        return map(lambda label: self.LabelScaler.label_to_uint(label), self.SupportedLabels)

