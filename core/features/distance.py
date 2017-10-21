import numpy as np
from feature import Feature
from core.source.relations import Relation


class DistanceFeature(Feature):

    def __init__(self):
        pass

    def create(self, relation):
        """ distance in chars between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        return [min(e1.end, e2.end) - max(e1.begin, e2.begin)]
