import logging
import numpy as np
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.labels.base import Label
from arekit.common.labels.pair import LabelPair
from arekit.contrib.networks.context.configurations.base.base import LabelCalculationMode
from arekit.networks.context.debug import DebugKeys
from arekit.networks.labeling.base import LabelsHelper


logger = logging.getLogger(__name__)


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
            logger.info([l.to_int() for l in text_opinion_labels])
            logger.info("Result: {}".format(label.to_int()))

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

        source = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Source)
        target = TextOpinionHelper.extract_entity_value(text_opinion, EntityEndType.Target)

        forward_opinion = Opinion(source_value=source,
                                  target_value=target,
                                  sentiment=label.Forward)

        backward_opinion = Opinion(source_value=target,
                                   target_value=source,
                                   sentiment=label.Backward)

        return [forward_opinion, backward_opinion]
