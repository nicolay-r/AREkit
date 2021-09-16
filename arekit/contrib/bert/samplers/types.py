from itertools import chain

from enum import Enum


class BertSampleProviderTypes(Enum):
    """
    Supported format types.
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

    __fmt_names = {
        BertSampleProviderTypes.CLASSIF_M: 'c_m',
        BertSampleProviderTypes.QA_M: "qa_m",
        BertSampleProviderTypes.NLI_M: 'nli_m',
        BertSampleProviderTypes.QA_B: "qa_b",
        BertSampleProviderTypes.NLI_B: "nli_b"
    }

    # region private methods

    @staticmethod
    def __iter_multiple():
        yield BertSampleProviderTypes.CLASSIF_M
        yield BertSampleProviderTypes.QA_M
        yield BertSampleProviderTypes.NLI_M

    @staticmethod
    def __iter_binary():
        yield BertSampleProviderTypes.QA_B
        yield BertSampleProviderTypes.NLI_B

    @staticmethod
    def __iter_all():
        return chain(SampleFormattersService.__iter_binary(),
                     SampleFormattersService.__iter_multiple())

    # endregion

    @staticmethod
    def is_binary(formatter_type):
        binary = list(BertSampleProviderTypes.__iter_binary())
        return formatter_type in binary

    @staticmethod
    def is_multiple(formatter_type):
        multiple = list(BertSampleProviderTypes.__iter_multiple())
        return formatter_type in multiple

    @staticmethod
    def iter_supported_names(return_values=False):
        for fmt_type in SampleFormattersService.__iter_all():
            yield SampleFormattersService.__fmt_names[fmt_type] if return_values else fmt_type

    @staticmethod
    def type_to_name(fmt_type):
        return SampleFormattersService.__fmt_names[fmt_type]

    @staticmethod
    def find_fmt_type_by_name(name):
        for fmt_type, fmt_type_name in SampleFormattersService.__fmt_names.items():
            if fmt_type_name == name:
                return fmt_type
        raise NotImplemented("Formatting type '{}' does not supported".format(name))