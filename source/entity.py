# -*- coding: utf-8 -*-
import io
from core.processing.lemmatization.base import Stemmer
from core.source.synonyms import SynonymsCollection


class EntityCollection:
    """ Collection of annotated entities
    """

    class KeyType:
        BY_SYNONYMS = 0
        BY_LEMMAS = 1
        BY_VALUE = 2

    def __init__(self, entities, stemmer, synonyms):
        """
        entities: list of Entity types
        stemmer: Stemmer
        """
        assert(isinstance(entities, list))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(synonyms, SynonymsCollection))

        entities.sort(key=lambda e: e.begin)

        self.entities = entities
        self.stemmer = stemmer
        self.synonyms = synonyms

        self.by_id = self.__create_index(entities, key_func=lambda e: e.ID)
        self.by_value = self.__create_index(entities, key_func=lambda e: e.value)
        self.by_lemmas = self.__create_index(
            entities, key_func=lambda e: stemmer.lemmatize_to_str(e.value))
        self.by_synonyms = self.__create_index(
            entities, key_func=lambda e: synonyms.get_synonym_group_index(e.value))

    @staticmethod
    def __create_index(entities, key_func):
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

    @classmethod
    def from_file(cls, filepath, stemmer, synonyms):
        """ Read annotation collection from file
        """
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(synonyms, SynonymsCollection))
        entities = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split()

                e_ID = args[0]
                e_str_type = args[1]
                e_begin = int(args[2])
                e_end = int(args[3])
                e_value = " ".join([a.strip().replace(',', '') for a in args[4:]])
                a = Entity(e_ID, e_str_type, e_begin, e_end, e_value)

                entities.append(a)

        return cls(entities, stemmer, synonyms)

    def get_entity_by_index(self, index):
        assert(isinstance(index, int))
        return self.entities[index]

    def get_entity_by_id(self, ID):
        assert(isinstance(ID, unicode))
        value = self.by_id[ID]
        assert(len(value) == 1)
        return value[0]

    def try_get_entities(self, value, group_key):
        assert(isinstance(value, unicode))
        if group_key == self.KeyType.BY_LEMMAS:
            key = self.stemmer.lemmatize_to_str(value)
            return self.__value_or_none(self.by_lemmas, key)
        if group_key == self.KeyType.BY_SYNONYMS:
            key = self.synonyms.get_synonym_group_index(value)
            return self.__value_or_none(self.by_synonyms, key)
        if group_key == self.KeyType.BY_VALUE:
            return self.__value_or_none(self.by_value, value)

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        for entity in self.entities:
            yield entity

# TODO. To /common/entity.py
class Entity:
    """ Entity description.
    """

    def __init__(self, ID, str_type, begin, end, value):
        assert(type(ID) == unicode)
        assert(type(str_type) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        assert(type(value) == unicode and len(value) > 0)
        self.ID = ID
        self.str_type = str_type
        self.begin = begin
        self.end = end
        self.value = value.lower()

    def get_int_ID(self):
        return int(self.ID[1:len(self.ID)])
