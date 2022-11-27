from arekit.common.entities.base import Entity
from arekit.common.news.entity import DocumentEntity
from arekit.common.news.parsed.base import ParsedNews


class BaseParsedNewsServiceProvider(object):

    def __init__(self, entity_index_func=None):
        """ Outside enity indexing function
            entity_index_func: provides id for a given entity, i.e.
                func(entity) -> int (id)
        """
        assert(callable(entity_index_func) or entity_index_func is None)
        self._doc_entities = None
        self.__entity_map = {}
        self.__entity_index_func = entity_index_func

    @property
    def Name(self):
        raise NotImplementedError()

    def init_parsed_news(self, parsed_news):
        assert(isinstance(parsed_news, ParsedNews))

        self._doc_entities = []
        self.__entity_map.clear()

        for index, entity in enumerate(parsed_news.iter_entities()):

            doc_entity = DocumentEntity(id_in_doc=index,
                                        value=entity.Value,
                                        e_type=entity.Type,
                                        display_value=entity.DisplayValue,
                                        group_index=entity.GroupIndex)

            self._doc_entities.append(doc_entity)

            if self.__entity_index_func is not None:
                self.__entity_map[self.__entity_index_func(entity)] = doc_entity

    def get_document_entity(self, entity):
        """ Maps entity to the related one with DocumentEntity type
        """
        assert(isinstance(entity, Entity))
        return self.__entity_map[self.__entity_index_func(entity)]

    def contains_entity(self, entity):
        return self.__entity_index_func(entity) in self.__entity_map
