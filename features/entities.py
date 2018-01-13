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

    def feature_names(self):
        return [self.__class__.__name__]


class EntityTagFeature(Base):

    tags = [u'PER', u'LOC', u'ORG', u'GEOPOLIT']

    def __init__(self):
        pass

    def create(self, relation):
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array(self._get_entity_tags(e1) + self._get_entity_tags(e2))

    def feature_names(self):
        class_name = self.__class__.__name__
        return [class_name + '_left_' + t for t in self.tags] + \
            [class_name + '_right_' + t for t in self.tags]

    def _get_entity_tags(self, entity):
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

    # TODO. simplify
    def feature_function_names(self):
        return [f + '_avg' for f in self.feature_names()]


class EntitySemanticClass(Base):

    def __init__(self, lowercase_words, semantic_class_name):
        assert(type(lowercase_words) == list)
        self.lowercase_words = lowercase_words
        self.semantic_class_name = semantic_class_name

    def create(self, relation):
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array([self._is_in_class(e1),  self._is_in_class(e2)])

    def feature_names(self):
        class_name = self.__class__.__name__ + '_' + self.semantic_class_name
        return [class_name + '_e1', class_name + '_e2']

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

    # TODO. simplify
    def feature_function_names(self):
        return [f + '_avg' for f in self.feature_names()]
