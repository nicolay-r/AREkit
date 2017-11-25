import numpy as np

from core.relations import Relation
from core.source.synonyms import SynonymsCollection
from core.source.entity import Entity
from base import Base


class EntitiesFrequency(Base):

    def __init__(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        self.synonyms = synonyms

    def create(self, relation):
        """ Frequency of entity occurance (and related synonyms) among all founded entities.
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        entities = relation.news.get_entities()

        e1_c = float(self._calculate_synonym_entities_count(e1))
        e2_c = float(self._calculate_synonym_entities_count(e2))

        e1_freq = e1_c / entities.count()
        e2_freq = e2_c / entities.count()

        return np.array([e1_freq, e2_freq])

    def _calculate_synonym_entities_count(self, e):
        assert(isinstance(e, Entity))
        e_c = 1
        if self.synonyms.has_synonym(e.value):
            e_c = len(self.synonyms.get_synonyms_list(e.value))
        return e_c
