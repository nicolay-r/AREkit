# -*- coding: utf-8 -*-
from arekit.bert.formatters.sample.base import BaseSampleFormatter
from arekit.bert.providers.label.binary import BertBinaryLabelProvider
from arekit.bert.providers.text.pair import PairTextProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert_samplers.utils import default_labels_formatter


class NliBinarySampleFormatter(BaseSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Binary variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, data_type, label_scaler, labels_formatter=None):
        assert(isinstance(labels_formatter, StringLabelsFormatter) or labels_formatter is None)

        text_b_template = u' {subject} к {object} в контексте << {context} >> -- {label}'
        formatter = default_labels_formatter() if labels_formatter is None else labels_formatter
        super(NliBinarySampleFormatter, self).__init__(
            data_type=data_type,
            text_provider=PairTextProvider(text_b_template=text_b_template,
                                           labels_formatter=formatter),
            label_provider=BertBinaryLabelProvider(label_scaler=label_scaler))
