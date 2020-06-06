# -*- coding: utf-8 -*-
import utils
from arekit.bert.formatters.sample.base import BaseSampleFormatter
from arekit.bert.providers.label.multiple import BertMultipleLabelProvider
from arekit.bert.providers.text.pair import PairTextProvider
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.synonyms import SynonymsCollection


class QaMultipleSampleFormatter(BaseSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, data_type, label_scaler, synonyms, labels_formatter=None, entity_formatter=None):
        assert(isinstance(labels_formatter, StringLabelsFormatter) or labels_formatter is None)
        assert(isinstance(entity_formatter, StringEntitiesFormatter) or entity_formatter is None)
        assert(isinstance(synonyms, SynonymsCollection))

        text_b_template = u'Что вы думаете по поводу отношения {subject} к {object} в контексте : << {context} >> ?'
        super(QaMultipleSampleFormatter, self).__init__(
            data_type=data_type,
            text_provider=PairTextProvider(
                text_b_template=text_b_template,
                synonyms=synonyms,
                labels_formatter=utils.default_labels_formatter() if labels_formatter is None else labels_formatter,
                entities_formatter=utils.default_entities_formatter() if entity_formatter is None else entity_formatter),
            label_provider=BertMultipleLabelProvider(label_scaler=label_scaler))
