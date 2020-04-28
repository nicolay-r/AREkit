from arekit.contrib.bert.formatters.sample.two_sentence import TwoSentenceSampleFormatter


class NliBSampleFormatter(TwoSentenceSampleFormatter):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)

    Binary variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    # TODO. Implement
    # TODO. Call three times rows filling (for each label).
