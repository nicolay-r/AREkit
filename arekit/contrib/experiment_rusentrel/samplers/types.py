from enum import Enum

from arekit.contrib.experiment_rusentrel.utils import EnumConversionService


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


class SampleFormattersService(EnumConversionService):

    _data = {
        'c_m': BertSampleProviderTypes.CLASSIF_M,
        "qa_m": BertSampleProviderTypes.QA_M,
        'nli_m': BertSampleProviderTypes.NLI_M,
        "qa_b": BertSampleProviderTypes.QA_B,
        "nli_b": BertSampleProviderTypes.NLI_B
    }
