import numpy as np
from feature import Feature
from core.source.relations import Relation


class EntitiesFrequency(Feature):

    def __init__(self):
        pass

    def create(self, relation):
        """ distance in chars between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)

        entities = relation.news.get_entities()
        e1_freq = (1.0*len(entities.get_by_value(e1.value)))/entities.count()
        e2_freq = (1.0*len(entities.get_by_value(e2.value)))/entities.count()

        return [e1_freq, e2_freq]
