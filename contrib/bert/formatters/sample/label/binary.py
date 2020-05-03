from arekit.common.labels.base import Label
from arekit.contrib.bert.formatters.sample.label.base import LabelProvider


class BinaryLabelProvider(LabelProvider):

    @staticmethod
    def get_label(expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        assert(isinstance(etalon_label, Label))
        return 1 if expected_label == etalon_label else 0

    def get_supported_labels(self):
        return [0, 1]
