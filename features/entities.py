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
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array([abs(e1.get_int_ID() - e2.get_int_ID()) - 1])


class EntityTagFeature(Base):

    tags = [u'PER', u'LOC', u'ORG', u'GEOPOLIT']

    def __init__(self):
        pass

    def create(self, relation):
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array(self._get_entity_tags(e1) + self._get_entity_tags(e2))

    def _get_entity_tags(self, entity):
        if entity.str_type not in self.tags:
            print entity.str_type
        assert(entity.str_type in self.tags)
        ind = self.tags.index(entity.str_type)
        result = [0] * len(self.tags)
        result[ind] = 1
        return result

    # TODO. simplify
    def calculate(self, relations):
        assert(type(relations) == list)

        results = []
        for relation in relations:
            results.append(self.create(relation))

        # No need to normalize.
        return np.average(results, axis=0)


class EntitySemanticClass(Base):

    def __init__(self, lowercase_words):
        assert(type(lowercase_words) == list)
        self.lowercase_words = lowercase_words

    def create(self, relation):
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array([self._is_in_class(e1),  self._is_in_class(e2)])

    def _is_in_class(self, entity):
        return 1 if entity.value in self.lowercase_words else 0

    # TODO. simplify
    def calculate(self, relations):
        assert(type(relations) == list)

        results = []
        for relation in relations:
            results.append(self.create(relation))

        # No need to normalize.
        return np.average(results, axis=0)
