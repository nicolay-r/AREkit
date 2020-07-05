# -*- coding: utf-8 -*-
from arekit.common.experiment.input.formatters.sample.base import BaseSampleFormatter
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.bert.terms.mapper import BertStringTextTermsMapper


def create_simple_sample_formatter(data_type, label_scaler, synonyms, entity_formatter):
    return BaseSampleFormatter(
        data_type=data_type,
        label_provider=MultipleLabelProvider(label_scaler=label_scaler),
        text_provider=BaseSingleTextProvider(create_default_terms_mapper(synonyms=synonyms,
                                                                         entity_formatter=entity_formatter)))


def create_default_terms_mapper(entity_formatter, synonyms):
    assert(isinstance(entity_formatter, StringEntitiesFormatter))
    assert(isinstance(synonyms, SynonymsCollection))
    return BertStringTextTermsMapper(entity_formatter=entity_formatter,
                                     synonyms=synonyms)


