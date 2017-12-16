import numpy as np
from base import Base


class EntityAppearanceFeature(Base):

    def __init__(self):
        pass

    def create(self, relation):
        """ check that e1 occurs before e2
        """
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array([e1.get_int_ID() < e2.get_int_ID()])

    def feature_names(self):
        return [self.__class__.__name__]
