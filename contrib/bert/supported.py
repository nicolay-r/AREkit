from enum import Enum


class BertSampleFormatter(Enum):
    """
    Supported formats
    """

    """
    Default formatter
    """
    CLASSIF_M = 0

    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = 1
    NLI_M = 2

    QA_B = 3
    NLI_B = 4


class SampleFormattersService(object):

    __to_str = {
        BertSampleFormatter.CLASSIF_M: u'c_m',
        BertSampleFormatter.QA_M: u"qa_m",
        BertSampleFormatter.NLI_M: u'nli_m',
        BertSampleFormatter.QA_B: u"qa_b",
        BertSampleFormatter.NLI_B: u"nli_b"
    }

    # region private methods

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
    def __iter_all():
        for formatter in SampleFormattersService.__iter_binary():
            yield formatter
        for formatter in SampleFormattersService.__iter_multiple():
            yield formatter

    # endregion

    @staticmethod
    def is_binary(formatter_type):
        binary = list(BertSampleFormatter.__iter_binary())
        return formatter_type in binary

    @staticmethod
    def is_multiple(formatter_type):
        multiple = list(BertSampleFormatter.__iter_multiple())
        return formatter_type in multiple

    @staticmethod
    def iter_supported(return_values=False):
        for fmt_type in SampleFormattersService.__iter_all():
            yield SampleFormattersService.__to_str[fmt_type] if return_values else fmt_type

    @staticmethod
    def type_to_value(fmt_type):
        return SampleFormattersService.__to_str[fmt_type]

    @staticmethod
    def find_type_by_value(value):
        for fmt_type, fmt_type_value in SampleFormattersService.__to_str.iteritems():
            if fmt_type_value == value:
                return fmt_type
        raise NotImplemented(u"Formatting type '{}' does not supported".format(value))