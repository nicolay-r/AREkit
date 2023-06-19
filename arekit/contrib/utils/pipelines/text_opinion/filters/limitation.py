from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class FrameworkLimitationsTextOpinionFilter(TextOpinionFilter):
    """ Note: this is an internal class, there is no need to
        adopt this from the outside of the AREkit.
        It is require to hide and provide known limitations.
    """

    def filter(self, text_opinion, parsed_doc, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_doc, ParsedDocument))

        s_ind = entity_service_provider.get_entity_position(
            text_opinion.SourceId, position_type=TermPositionTypes.SentenceIndex)
        t_ind = entity_service_provider.get_entity_position(
            text_opinion.TargetId, position_type=TermPositionTypes.SentenceIndex)

        if s_ind != t_ind:
            # AREkit does not provide a support for multi-sentence opinions at present.
            return False

        return True
