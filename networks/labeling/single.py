import numpy as np

from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion

from arekit.networks.context.configurations.base import LabelCalculationMode
from arekit.networks.labeling.base import LabelsHelper


class SingleLabelsHelper(LabelsHelper):

    @staticmethod
    def get_classes_count():
        return 3

    @staticmethod
    def create_label_from_text_opinions(text_opinion_labels, label_creation_mode):
        assert(isinstance(text_opinion_labels, list))
        assert(isinstance(label_creation_mode, unicode))

        label = None
        if label_creation_mode == LabelCalculationMode.FIRST_APPEARED:
            label = text_opinion_labels[0]
        if label_creation_mode == LabelCalculationMode.AVERAGE:
            forwards = [l.to_int() for l in text_opinion_labels]
            label = Label.from_int(np.sign(sum(forwards)))

        return label

    @staticmethod
    def create_label_from_uint(label_uint):
        assert(label_uint >= 0)
        return Label.from_uint(label_uint)

    @staticmethod
    def create_label_from_opinions(forward, backward):
        assert(isinstance(forward, Opinion))
        return forward.Sentiment

    @staticmethod
    def create_opinions_from_text_opinion_and_label(text_opinion, label):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(label, Label))

        source = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Source)
        target = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Target)

        opinion = Opinion(source_value=source,
                          target_value=target,
                          sentiment=label)

        return [opinion]

