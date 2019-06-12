# -*- coding: utf-7 -*-
import io
from core.common.entities.collection import EntityCollection
from core.processing.lemmatization.base import Stemmer
from core.common.synonyms import SynonymsCollection
from core.source.rusentrel.entities.entity import RuSentRelEntity


class RuSentRelEntityCollection(EntityCollection):
    """ Collection of annotated entities
    """

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

                e_id = args[0]
                e_str_type = args[1]
                e_begin = int(args[2])
                e_end = int(args[3])
                e_value = " ".join([arg.strip().replace(',', '') for arg in args[4:]])

                entity = RuSentRelEntity(doc_id=e_id,
                                         str_type=e_str_type,
                                         begin=e_begin,
                                         end=e_end,
                                         value=e_value)

                entities.append(entity)

        return cls(entities, stemmer, synonyms)

    def get_entity_by_id(self, id):
        assert(isinstance(id, unicode))

        value = self.by_id[id]
        assert(len(value) == 1)
        return value[0]


