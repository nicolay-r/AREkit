from arekit.common.labels.base import Label
from arekit.contrib.bert.providers.label.base import BertLabelProvider


class BertBinaryLabelProvider(BertLabelProvider):

    def calculate_output_label_uint(self, expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        assert(isinstance(etalon_label, Label))
        return 1 if expected_label == etalon_label else 0

    @property
    def OutputLabels(self):
        return [0, 1]