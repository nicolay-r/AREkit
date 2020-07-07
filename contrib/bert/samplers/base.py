# -*- coding: utf-8 -*-
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.terms_mapper import StringTextTermsMapper


def create_simple_sample_formatter(data_type, label_scaler, text_terms_mapper):
    assert(isinstance(text_terms_mapper, StringTextTermsMapper))

    return BaseSampleFormatter(
        data_type=data_type,
        label_provider=MultipleLabelProvider(label_scaler=label_scaler),
        text_provider=BaseSingleTextProvider(text_terms_mapper=text_terms_mapper))


