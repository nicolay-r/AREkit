import numpy as np

from core.relations import Relation
from base import Base


class EntitiesFrequency(Base):

    def __init__(self):
        pass

    def create(self, relation):
        """ distance in chars between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        entities = relation.news.get_entities()
        e1_freq = (1.0 * len(entities.get_entity_by_value(e1.value))) / entities.count()
        e2_freq = (1.0 * len(entities.get_entity_by_value(e2.value))) / entities.count()

        return np.array([e1_freq, e2_freq])
