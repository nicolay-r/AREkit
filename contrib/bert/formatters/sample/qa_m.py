# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.two_sentence import TwoSentenceSampleFormatter


class QaMSampleFormatter(TwoSentenceSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def get_text_template(self):
        return u'Что вы думаете по поводу отношения {subject} к {object} в контексте : " {context} " ?'
