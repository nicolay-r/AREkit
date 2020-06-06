# -*- coding: utf-8 -*-
import utils
from arekit.bert.formatters.sample.base import BaseSampleFormatter
from arekit.bert.providers.label.multiple import BertMultipleLabelProvider
from arekit.bert.providers.text.single import SingleTextProvider
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection


def create_simple_sample_formatter(data_type, label_scaler, synonyms, entity_formatter=None):
    assert(isinstance(entity_formatter, StringEntitiesFormatter) or entity_formatter is None)
    assert(isinstance(synonyms, SynonymsCollection))

    return BaseSampleFormatter(
        data_type=data_type,
        label_provider=BertMultipleLabelProvider(label_scaler=label_scaler),
        text_provider=SingleTextProvider(
            entities_formatter=utils.default_entities_formatter() if entity_formatter is None else entity_formatter,
            synonyms=synonyms))

