import numpy as np
from feature import Feature


class EntityAppearanceFeature(Feature):

    def __init__(self):
        pass

    def create(self, relation):
        """ check that e1 occurs before e2
        """
        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)
        return np.array([e1.get_int_ID() < e2.get_int_ID()])
