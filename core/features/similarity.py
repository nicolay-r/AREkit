# TODO: from w2v features

import numpy as np

from core.relations import Relation
from core.stemmer import Stemmer
from feature import Feature


class SimilarityFeature(Feature):

    def __init__(self, w2v_model):
        self.stemmer = Stemmer()
        self.w2v_model = w2v_model

    def __get_mean_word2vec_vector(self, lemmas):
        v = np.zeros(self.w2v_model.vector_size, dtype=np.float32)
        for l in lemmas:
            if l in self.w2v_model:
                v = v + self.w2v_model[l]
        return v

    def create(self, relation):
        """ Get the similarity between two entities of relation
        """
        assert(isinstance(relation, Relation))

        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)
        e1_value_lemmas = self.stemmer.lemmatize_to_rusvectores_str(e1.value)
        e2_value_lemmas = self.stemmer.lemmatize_to_rusvectores_str(e2.value)
        v1 = self.__get_mean_word2vec_vector(e1_value_lemmas)
        v2 = self.__get_mean_word2vec_vector(e2_value_lemmas)
        return np.array([sum(map(lambda x, y: x * y, v1, v2))])
