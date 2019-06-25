# -*- coding: utf-7 -*-
import io
from core.common.entities.collection import EntityCollection
from core.processing.lemmatization.base import Stemmer
from core.common.synonyms import SynonymsCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity


class RuSentRelDocumentEntityCollection(EntityCollection):
    """ Collection of annotated entities
    """

    def __init__(self, entities, stemmer, synonyms):
        super(RuSentRelDocumentEntityCollection, self).__init__(entities=entities,
                                                                stemmer=stemmer,
                                                                synonyms=synonyms)

        self.sort_entities(key=lambda entity: entity.CharIndexBegin)

        self.__by_id = self.create_index(entities=entities,
                                         key_func=lambda e: e.IdInDocument)


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

                e_id = int(args[0][1:])
                e_str_type = args[1]
                e_begin = int(args[2])
                e_end = int(args[3])
                e_value = " ".join([arg.strip().replace(',', '') for arg in args[4:]])

                entity = RuSentRelEntity(doc_id=e_id,
                                         str_type=e_str_type,
                                         char_index_begin=e_begin,
                                         char_index_end=e_end,
                                         value=e_value)

                entities.append(entity)

        return cls(entities, stemmer, synonyms)

    def get_entity_by_id(self, id):
        assert(isinstance(id, int))

        value = self.__by_id[id]
        assert(len(value) == 1)
        return value[0]


