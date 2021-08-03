# -*- coding: utf-8 -*-
from arekit.common.synonyms import SynonymsCollection


class EntityCollection(object):
    """ Collection of annotated entities
    """

    class KeyType:
        BY_SYNONYMS = 0
        BY_VALUE = 1

    def __init__(self, entities, synonyms):
        assert(isinstance(entities, list))
        assert(isinstance(synonyms, SynonymsCollection))

        self.__entities = entities
        self.__synonyms = synonyms

        self.__by_value = self.create_index(entities=entities,
                                            key_func=lambda e: e.Value)

        self.__by_synonyms = self.create_index(
            entities=entities,
            key_func=lambda e: synonyms.get_synonym_group_index(e.Value))

        self.__by_id = self.create_index(entities=entities,
                                         key_func=lambda e: e.IdInDocument)

    def sort_entities(self, key):
        assert(callable(key))
        self.__entities.sort(key=key)

    @staticmethod
    def create_index(entities, key_func):
        index = {}
        for e in entities:
            key = key_func(e)
            if key in index:
                index[key].append(e)
            else:
                index[key] = [e]
        return index

    @staticmethod
    def __value_or_none(d, key):
        return d[key] if key in d else None

    def get_entity_by_index(self, index):
        assert(isinstance(index, int))
        return self.__entities[index]

    def try_get_entities(self, value, group_key):
        assert(isinstance(value, unicode))

        if group_key == self.KeyType.BY_SYNONYMS:
            key = self.__synonyms.get_synonym_group_index(value)
            return self.__value_or_none(self.__by_synonyms, key)
        if group_key == self.KeyType.BY_VALUE:
            return self.__value_or_none(self.__by_value, value)

    def get_entity_by_id(self, id):
        assert(isinstance(id, int))

        value = self.__by_id[id]
        assert(len(value) == 1)
        return value[0]

    def __len__(self):
        return len(self.__entities)

    def __iter__(self):
        for entity in self.__entities:
            yield entity
