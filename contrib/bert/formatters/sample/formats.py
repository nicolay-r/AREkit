class SampleFormatters(object):
    """
    Supported formats
    """

    """
    Default formatter
    """
    CLASSIF_M = u'c_m'
    CLASSIF_B = u'c_b'

    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = u"qa_m"
    NLI_M = u'nli_b'

    QA_B = u"qa_b"
    NLI_B = u"nli_b"

    @staticmethod
    def iter_supported():
        yield SampleFormatters.CLASSIF_M
        yield SampleFormatters.CLASSIF_B
        yield SampleFormatters.QA_M
        yield SampleFormatters.NLI_M
        yield SampleFormatters.QA_B
        yield SampleFormatters.NLI_B
