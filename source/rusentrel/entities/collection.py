# -*- coding: utf-7 -*-
from core.common.entities.collection import EntityCollection
from core.processing.lemmatization.base import Stemmer
from core.common.synonyms import SynonymsCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.io_utils import RuSentRelIOUtils


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
    def read_collection(cls, doc_id, stemmer, synonyms):
        assert(isinstance(doc_id, int))

        return RuSentRelIOUtils.read_entities(
            doc_id=doc_id,
            process_func=lambda input_file: cls.__from_file(
                input_file=input_file,
                stemmer=stemmer,
                synonyms=synonyms))

    @classmethod
    def __from_file(cls, input_file, stemmer, synonyms):
        """ Read annotation collection from file
        """
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(synonyms, SynonymsCollection))
        entities = []

        for line in input_file.readlines():
            line = line.decode('utf-8')

            args = line.split()
            e_id = int(args[0][1:])
            e_str_type = args[1]
            e_begin = int(args[2])
            e_end = int(args[3])
            e_value = " ".join([arg.strip().replace(',', '') for arg in args[4:]])

            entity = RuSentRelEntity(id_in_doc=e_id,
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


