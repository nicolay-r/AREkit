from arekit.common.experiment.input.providers.label.base import LabelProvider
from arekit.common.labels.base import Label


class BinaryLabelProvider(LabelProvider):

    def calculate_output_label(self, expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        assert(isinstance(etalon_label, Label))
        return 1 if expected_label == etalon_label else 0

    @property
    def OutputLabelsUint(self):
        return [0, 1]
