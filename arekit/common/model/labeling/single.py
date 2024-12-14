from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.labeling.modes import LabelCalculationMode


class SingleLabelsHelper(LabelsHelper):

    @staticmethod
    def __sign(x):
        if x == 0:
            return 0
        return -1 if x < 0 else 1

    def aggregate_labels(self, labels_list, label_calc_mode):
        assert(isinstance(labels_list, list))
        assert(isinstance(label_calc_mode, LabelCalculationMode))

        label = None

        if label_calc_mode == LabelCalculationMode.FIRST_APPEARED:
            label = labels_list[0]

        if label_calc_mode == LabelCalculationMode.AVERAGE:
            int_labels = [self._label_scaler.label_to_int(label)
                          for label in labels_list]
            label = self._label_scaler.int_to_label(SingleLabelsHelper.__sign(sum(int_labels)))

        return label

