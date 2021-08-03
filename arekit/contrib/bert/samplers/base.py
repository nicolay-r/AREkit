# -*- coding: utf-8 -*-
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.terms_mapper import OpinionContainingTextTermsMapper


def create_simple_sample_formatter(data_type, label_scaler, text_terms_mapper, balance):
    assert(isinstance(text_terms_mapper, OpinionContainingTextTermsMapper))

    return BaseSampleFormatter(
        data_type=data_type,
        label_provider=MultipleLabelProvider(label_scaler=label_scaler),
        text_provider=BaseSingleTextProvider(text_terms_mapper=text_terms_mapper),
        balance=data_type == DataType.Train and balance)
