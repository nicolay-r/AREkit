from arekit.common.news.entity import DocumentEntity
from arekit.common.news.parsed.base import ParsedNews


class BaseParsedNewsServiceProvider(object):

    def __init__(self):
        self._doc_entities = None

    @property
    def Name(self):
        raise NotImplementedError()

    def init_parsed_news(self, parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        self._doc_entities = {}
        for entity in parsed_news.iter_entities():
            assert(entity.ID not in self._doc_entities)
            self._doc_entities[entity.ID] = DocumentEntity(id_in_doc=entity.ID,
                                                           value=entity.Value,
                                                           e_type=entity.Type,
                                                           group_index=entity.GroupIndex)
