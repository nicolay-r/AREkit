import numpy as np

from arekit.common.labels.base import Label
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion


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

    @staticmethod
    def iter_opinions_from_text_opinion_and_label(text_opinion, label):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(label, Label))

        source = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Source)
        target = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Target)

        yield Opinion(source_value=source,
                      target_value=target,
                      sentiment=label)

