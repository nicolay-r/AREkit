from arekit.common.labels.base import Label
from arekit.contrib.bert.formatters.sample.label.base import LabelProvider


class MultipleLabelProvider(LabelProvider):

    def __init__(self, supported_labels):
        super(MultipleLabelProvider, self).__init__(supported_labels=supported_labels)

    @staticmethod
    def get_label(expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        return expected_label.to_uint()

    def get_supported_labels(self):
        return self.SupportedLabels

