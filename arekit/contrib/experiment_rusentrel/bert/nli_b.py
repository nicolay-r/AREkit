from arekit.common.data.input.providers.label.binary import BinaryLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider


class NliBinarySampleProvider(BaseSampleRowProvider):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Binary variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, label_scaler, text_b_labels_fmt, text_terms_mapper):
        assert(isinstance(text_b_labels_fmt, StringLabelsFormatter))
        assert(isinstance(text_terms_mapper, OpinionContainingTextTermsMapper))

        text_b_template = '{subject} к {object} в контексте << {context} >> -- {label}'
        super(NliBinarySampleProvider, self).__init__(
            text_provider=PairTextProvider(
                text_b_template=text_b_template,
                text_b_labels_fmt=text_b_labels_fmt,
                text_terms_mapper=text_terms_mapper),
            label_provider=BinaryLabelProvider(label_scaler=label_scaler))
