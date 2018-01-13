import numpy as np

from base import Base
from core.source.lexicon import RelationLexicon
from core.source.synonyms import SynonymsCollection
from core.relations import Relation


class ExternalRelationsFeature(Base):

    def __init__(self, relation_lexicon, synonyms):
        """ Relation lexicon: a dictionary, where each value has a following
            template: X <-> X, where X is a str.
        """
        assert(isinstance(relation_lexicon, RelationLexicon))
        assert(isinstance(synonyms, SynonymsCollection))
        self.relation_lexicon = relation_lexicon
        self.synonyms = synonyms

    def calculate(self, relations):
        """ functions_list: np.average
        """
        assert(type(relations) == list)
        results = []
        for relation in relations:
            results.append(self.create(relation))
        return self._normalize(
                np.array([np.average(results)])
            )

    def create(self, relation):
        """ Get the similarity between two entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        return np.array([self.calculate_score(e1.value, e2.value)])

    def calculate_score(self, e1_value, e2_value):
        assert(type(e1_value) == unicode)
        assert(type(e2_value) == unicode)

        default_score = 0

        if not (self.synonyms.has_synonym(e1_value) and self.synonyms.has_synonym(e2_value)):
            return default_score

        for s1 in self.synonyms.get_synonyms_list(e1_value):
            for s2 in self.synonyms.get_synonyms_list(e2_value):
                score = self.relation_lexicon.get_score(s1, s2)
                if score is not None:
                    return score

        return default_score

    def feature_names(self):
        return [self.__class__.__name__]
