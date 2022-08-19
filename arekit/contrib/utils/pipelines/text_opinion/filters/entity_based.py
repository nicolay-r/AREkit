from arekit.common.entities.types import OpinionEntityType
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class EntityBasedTextOpinionFilter(TextOpinionFilter):

    def __init__(self, entity_filter):
        super(EntityBasedTextOpinionFilter, self).__init__()
        self.__entity_filter = entity_filter

    def filter(self, text_opinion, parsed_news, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_news, ParsedNews))

        e_source = entity_service_provider._doc_entities[text_opinion.SourceId]
        if self.__entity_filter is not None and self.__entity_filter.is_ignored(e_source, OpinionEntityType.Subject):
            return False

        e_target = entity_service_provider._doc_entities[text_opinion.TargetId]
        if self.__entity_filter is not None and self.__entity_filter.is_ignored(e_target, OpinionEntityType.Object):
            return False

        return True
