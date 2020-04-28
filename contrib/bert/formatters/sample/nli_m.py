# -*- coding: utf-8 -*-
from arekit.contrib.bert.formatters.sample.two_sentence import TwoSentenceSampleFormatter


class NliMSampleFormatter(TwoSentenceSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    def get_text_template(self):
        return u' {subject} к {object} в контексте : " {context} "'
