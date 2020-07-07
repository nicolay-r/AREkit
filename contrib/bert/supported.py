from enum import Enum


class BertSampleFormatter(Enum):
    """
    Supported formats
    """

    """
    Default formatter
    """
    CLASSIF_M = u'c_m'

    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = u"qa_m"
    NLI_M = u'nli_m'

    QA_B = u"qa_b"
    NLI_B = u"nli_b"


class SampleFormattersService(object):

    @staticmethod
    def __iter_multiple():
        yield BertSampleFormatter.CLASSIF_M
        yield BertSampleFormatter.QA_M
        yield BertSampleFormatter.NLI_M

    @staticmethod
    def __iter_binary():
        yield BertSampleFormatter.QA_B
        yield BertSampleFormatter.NLI_B

    @staticmethod
    def is_binary(formatter_type):
        binary = list(BertSampleFormatter.__iter_binary())
        return formatter_type in binary

    @staticmethod
    def is_multiple(formatter_type):
        multiple = list(BertSampleFormatter.__iter_multiple())
        return formatter_type in multiple

    @staticmethod
    def iter_supported():
        for formatter in SampleFormattersService.__iter_binary():
            yield formatter
        for formatter in SampleFormattersService.__iter_multiple():
            yield formatter
