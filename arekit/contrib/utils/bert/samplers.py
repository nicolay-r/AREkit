from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider


def create_sample_provider(label_scaler, text_b_template, text_terms_mapper, text_b_labels_fmt=None):
    assert(isinstance(text_terms_mapper, OpinionContainingTextTermsMapper))

    text_provider = BaseSingleTextProvider(text_terms_mapper=text_terms_mapper) \
        if text_b_labels_fmt is None else PairTextProvider(text_b_template=text_b_template,
                                                           text_b_labels_fmt=text_b_labels_fmt,
                                                           text_terms_mapper=text_terms_mapper)

    label_provider = MultipleLabelProvider(label_scaler=label_scaler)

    return BaseSampleRowProvider(text_provider=text_provider, label_provider=label_provider)
