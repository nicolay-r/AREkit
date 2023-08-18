from collections import Counter

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

        c = Counter()
        for _, entity in enumerate(parsed_news.iter_entities()):
            assert(isinstance(entity, Entity))

            # Register childrens.
            doc_childs = {}
            for ce in entity.iter_childs():
                de = DocumentEntity(id_in_doc=c["entities"],
                                    value=ce.Value,
                                    e_type=ce.Type,
                                    childs=None,
                                    display_value=ce.DisplayValue,
                                    group_index=ce.GroupIndex)
                c["entities"] += 1
                doc_childs[ce] = de

                self._doc_entities.append(de)

            # Register Root node.
            doc_entity = DocumentEntity(id_in_doc=c["entities"],
                                        value=entity.Value,
                                        e_type=entity.Type,
                                        childs=list(doc_childs.values()) if len(doc_childs) > 0 else None,
                                        display_value=entity.DisplayValue,
                                        group_index=entity.GroupIndex)
            c["entities"] += 1

            self._doc_entities.append(doc_entity)

            if self.__entity_index_func is not None:
                # For root node.
                assert(self.__entity_index_func(entity) not in self.__entity_map)
                self.__entity_map[self.__entity_index_func(entity)] = doc_entity
                # For children.
                for ce, de in doc_childs.items():
                    assert(self.__entity_index_func(ce) not in self.__entity_map)
                    self.__entity_map[self.__entity_index_func(ce)] = de

        #print("Document entites registred:", len(self._doc_entities))

    def get_document_entity(self, entity):
        """ Maps entity to the related one with DocumentEntity type
        """
        assert(isinstance(entity, Entity))
        return self.__entity_map[self.__entity_index_func(entity)]

    def contains_entity(self, entity):
        return self.__entity_index_func(entity) in self.__entity_map
