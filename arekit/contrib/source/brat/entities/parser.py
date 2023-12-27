from arekit.common.docs.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.text.partitioning.str import StringPartitioning
from arekit.common.text.partitioning.terms import TermsPartitioning
from arekit.contrib.source.brat.sentence import BratSentence


class BratTextEntitiesParser(SentenceObjectsParserPipelineItem):

    ################################
    # NOTE: Supported partitionings.
    ################################
    # By default, BRAT annotation proposes to adopt entities annotation
    # based on string input, which means that entity ends described as
    # `char-ind-begin` and `char-ind-end`. However, the latter could be
    # expanded to list of terms, which means that we deal with `ind-begin`
    # and `ind-end` list indices.
    __supported_partitionings = {
        "string": StringPartitioning(),
        "terms": TermsPartitioning()
    }

    def __init__(self, partitioning="string", **kwargs):
        assert(isinstance(partitioning, str))
        super(BratTextEntitiesParser, self).__init__(self.__supported_partitionings[partitioning], **kwargs)

    # region protected methods

    def _get_text(self, sentence):
        return sentence.Text

    def _get_parts_provider_func(self, sentence):
        return self.__iter_subs_values_with_bounds(sentence)

    # endregion

    # region private methods

    @staticmethod
    def __iter_subs_values_with_bounds(sentence):
        assert(isinstance(sentence, BratSentence))
        return sentence.iter_entity_with_local_bounds()

    # endregion