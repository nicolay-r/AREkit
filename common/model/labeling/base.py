from arekit.common.experiment.scales.base import BaseLabelScaler


class LabelsHelper(object):

    def __init__(self, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))
        self._label_scaler = label_scaler

    def label_from_uint(self, value):
        return self._label_scaler.uint_to_label(value=value)

    def label_to_uint(self, label):
        return self._label_scaler.label_to_uint(value=label)

    def get_classes_count(self):
        return len(self._label_scaler.ordered_suppoted_labels())

    def aggregate_labels(self, labels_list, label_creation_mode):
        raise NotImplementedError()

    @staticmethod
    def iter_opinions_from_text_opinion_and_label(text_opinion, label):
        raise NotImplementedError()
