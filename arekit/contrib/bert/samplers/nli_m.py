from arekit.common.experiment.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider


class NliMultipleSampleProvider(BaseSampleRowProvider):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, label_scaler, labels_formatter, text_terms_mapper):
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(text_terms_mapper, OpinionContainingTextTermsMapper))

        text_b_template = '{subject} к {object} в контексте : << {context} >>'
        super(NliMultipleSampleProvider, self).__init__(
            text_provider=PairTextProvider(
                text_b_template=text_b_template,
                labels_formatter=labels_formatter,
                text_terms_mapper=text_terms_mapper),
            label_provider=MultipleLabelProvider(label_scaler=label_scaler))
