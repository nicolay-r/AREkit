import numpy as np

from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.labeling.modes import LabelCalculationMode


class SingleLabelsHelper(LabelsHelper):

    def aggregate_labels(self, labels_list, label_creation_mode):
        assert(isinstance(labels_list, list))
        assert(isinstance(label_creation_mode, unicode))
        assert(LabelCalculationMode.supported(label_creation_mode))

        label = None

        if label_creation_mode == LabelCalculationMode.FIRST_APPEARED:
            label = labels_list[0]

        if label_creation_mode == LabelCalculationMode.AVERAGE:
            int_labels = [self._label_scaler.label_to_int(label)
                          for label in labels_list]
            label = self._label_scaler.int_to_label(np.sign(sum(int_labels)))

        return label

