import numpy as np

from core.relations import Relation
from base import Base


class EntitiesBetweenFeature(Base):

    def __init__(self):
        pass

    def create(self, relation):
        """ Get entities count between relation entities
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)
        return np.array([abs(e1.get_int_ID() - e2.get_int_ID()) - 1])
