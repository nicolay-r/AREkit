from arekit.common.labels.base import Label


class MultipleLabelProvider(object):

    @staticmethod
    def get_label(expected_label, etalon_label):
        assert(isinstance(expected_label, Label))
        return expected_label.to_uint()

    @staticmethod
    def get_supported_labels():
        return Label._get_supported_labels()

