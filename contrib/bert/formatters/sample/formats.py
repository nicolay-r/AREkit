class SampleFormatters(object):
    """
    Supported formats
    """

    """
    Default formatter
    """
    DEFAULT = u'default'

    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = u"qa_m"
    NLI_M = u'nli_b'

    @staticmethod
    def iter_supported():
        yield SampleFormatters.DEFAULT
        yield SampleFormatters.QA_M
        yield SampleFormatters.NLI_M
