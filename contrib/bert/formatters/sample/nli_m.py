# -*- coding: utf-8 -*-
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter
from arekit.contrib.bert.formatters.str_label_fmt import RussianThreeScaleLabelsFormatter
from arekit.contrib.bert.providers.label.multiple import BertMultipleLabelProvider
from arekit.contrib.bert.providers.text.pair import PairTextProvider


class NliMultipleSampleFormatter(BaseSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, data_type, label_scaler, synonyms, entities_formatter):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(entities_formatter, StringEntitiesFormatter))

        text_b_template = u'субъект к объекту в контексте : << {context} >>'
        super(NliMultipleSampleFormatter, self).__init__(
            data_type=data_type,
            text_provider=PairTextProvider(text_b_template=text_b_template,
                                           labels_formatter=RussianThreeScaleLabelsFormatter(),
                                           entities_formatter=entities_formatter,
                                           synonyms=synonyms),
            label_provider=BertMultipleLabelProvider(label_scaler=label_scaler))
