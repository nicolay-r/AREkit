class SampleFormatters(object):
    """
    Supported formats
    """

    """
    Default formatter
    """
    COLA = 'cola'

    """
    Natural Language Inference samplers
    paper:
    """
    QA_M = "qa_m"
    NLI_M = 'nli_b'

    @staticmethod
    def iter_supported():
        yield SampleFormatters.COLA
        yield SampleFormatters.QA_M
        yield SampleFormatters.NLI_M
