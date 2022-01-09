from arekit.common.labels.scaler.base import BaseLabelScaler


class LabelsHelper(object):

    def __init__(self, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))
        self._label_scaler = label_scaler

    def label_from_uint(self, value):
        return self._label_scaler.uint_to_label(value=value)

    def label_to_uint(self, label):
        return self._label_scaler.label_to_uint(label=label)

    def get_classes_count(self):
        return len(self._label_scaler.ordered_suppoted_labels())

    def aggregate_labels(self, labels_list, label_calc_mode):
        raise NotImplementedError()

    @staticmethod
    def compose_opinion(text_opinion, label):
        raise NotImplementedError()
