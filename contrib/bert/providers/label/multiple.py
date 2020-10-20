from arekit.common.labels.base import Label
from arekit.contrib.bert.providers.label.base import BertLabelProvider


class BertMultipleLabelProvider(BertLabelProvider):

    def __init__(self, label_scaler):
        super(BertMultipleLabelProvider, self).__init__(label_scaler=label_scaler)

    def calculate_output_label_uint(self, expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        return self.LabelScaler.label_to_uint(label=expected_label)

    @property
    def OutputLabels(self):
        return self.SupportedLabels
