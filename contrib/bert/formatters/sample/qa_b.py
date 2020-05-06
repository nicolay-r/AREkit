# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter

from arekit.contrib.bert.formatters.sample.label.binary import BertBinaryLabelProvider
from arekit.contrib.bert.formatters.sample.text.pair import PairTextProvider


class QaBinarySampleFormatter(BaseSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def __init__(self, data_type, label_scaler):

        text_b_template = u'Отношение {subject} к {object} в контексте " {context} " -- {label} ?'
        super(QaBinarySampleFormatter, self).__init__(
            data_type=data_type,
            text_provider=PairTextProvider(text_b_template),
            label_provider=BertBinaryLabelProvider(label_scaler=label_scaler))
