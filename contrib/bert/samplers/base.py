# -*- coding: utf-8 -*-
from arekit.common.experiment.input.formatters.sample.base import BaseSampleFormatter
from arekit.common.experiment.input.providers.label.multiple import BertMultipleLabelProvider
from arekit.common.experiment.input.providers.text.single import SingleTextProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection


def create_simple_sample_formatter(data_type, label_scaler, synonyms, entity_formatter):
    assert(isinstance(entity_formatter, StringEntitiesFormatter))
    assert(isinstance(synonyms, SynonymsCollection))

    return BaseSampleFormatter(
        data_type=data_type,
        label_provider=BertMultipleLabelProvider(label_scaler=label_scaler),
        text_provider=SingleTextProvider(
            entities_formatter=entity_formatter,
            synonyms=synonyms))

