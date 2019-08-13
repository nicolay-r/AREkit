import numpy as np
from core.common.text_opinions.helper import TextOpinionHelper
from core.common.text_opinions.text_opinion import TextOpinion
from core.common.opinions.opinion import Opinion
from core.common.text_opinions.end_type import EntityEndType

from core.evaluation.labels import Label, LabelPair

from core.networks.context.configurations.base import LabelCalculationMode
from core.networks.context.debug import DebugKeys
from core.networks.labeling.base import LabelsHelper


class PairedLabelsHelper(LabelsHelper):

    @staticmethod
    def get_classes_count():
        return 9

    @staticmethod
    def create_label_from_uint(label_uint):
        assert(label_uint >= 0)
        return LabelPair.from_uint(label_uint)

    @staticmethod
    def create_label_from_text_opinions(text_opinion_labels, label_creation_mode):
        assert(isinstance(text_opinion_labels, list))
        assert(isinstance(label_creation_mode, unicode))

        label = None
        if label_creation_mode == LabelCalculationMode.FIRST_APPEARED:
            label = text_opinion_labels[0]
        if label_creation_mode == LabelCalculationMode.AVERAGE:
            forwards = [l.Forward.to_int() for l in text_opinion_labels]
            backwards = [l.Backward.to_int() for l in text_opinion_labels]
            label = LabelPair(forward=Label.from_int(np.sign(sum(forwards))),
                              backward=Label.from_int(np.sign(sum(backwards))))

        if DebugKeys.PredictLabel:
            print [l.to_int() for l in text_opinion_labels]
            print "Result: {}".format(label.to_int())

        # TODO: Correct label

        return label

    @staticmethod
    def create_label_from_opinions(forward, backward):
        assert(isinstance(forward, Opinion))
        assert(isinstance(backward, Opinion))
        return LabelPair(forward=forward.Sentiment,
                         backward=backward.Sentiment)

    @staticmethod
    def create_opinions_from_text_opinion_and_label(text_opinion, label):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(label, LabelPair))

        source = TextOpinionHelper.EntityValue(text_opinion, EntityEndType.Source)
        target = TextOpinionHelper.EntityValue(text_opinion, EntityEndType.Target)

        forward_opinion = Opinion(source_value=source,
                                  target_value=target,
                                  sentiment=label.Forward)

        backward_opinion = Opinion(source_value=target,
                                   target_value=source,
                                   sentiment=label.Backward)

        return [forward_opinion, backward_opinion]
