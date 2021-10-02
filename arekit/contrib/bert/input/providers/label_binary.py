from arekit.common.data.input.providers.label.base import LabelProvider


class BinaryLabelProvider(LabelProvider):

    def calculate_output_uint_label(self, expected_uint_label, etalon_uint_label):
        return 1 if expected_uint_label == etalon_uint_label else 0

    @property
    def OutputLabelsUint(self):
        return [0, 1]
