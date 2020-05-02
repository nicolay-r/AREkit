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
    def __iter_multiple():
        yield SampleFormatters.QA_M
        yield SampleFormatters.NLI_M
        yield SampleFormatters.CLASSIF_M

    @staticmethod
    def __iter_binary():
        yield SampleFormatters.CLASSIF_B
        yield SampleFormatters.QA_B
        yield SampleFormatters.NLI_B

    @staticmethod
    def is_binary(formatter_type):
        binary = list(SampleFormatters.__iter_binary())
        return formatter_type in binary

    @staticmethod
    def is_multiple(formatter_type):
        multiple = list(SampleFormatters.__iter_multiple())
        return formatter_type in multiple

    @staticmethod
    def iter_supported():
        for formatter in SampleFormatters.__iter_binary():
            yield formatter
        for formatter in SampleFormatters.__iter_multiple():
            yield formatter
