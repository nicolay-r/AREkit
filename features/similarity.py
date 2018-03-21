# TODO: from w2v features

import numpy as np

from base import Base
from core.processing.stemmer import Stemmer
from core.runtime.relations import Relation
from core.runtime.embeddings import Embedding


class SimilarityFeature(Base):

    def __init__(self, embedding):
        assert(isinstance(embedding, Embedding))
        self.stemmer = Stemmer()
        self.embedding = embedding

    def __get_mean_word2vec_vector(self, lemmas):
        v = np.zeros(self.embedding.vector_size, dtype=np.float32)
        for l in lemmas:
            if l in self.embedding:
                v = v + self.embedding[l]
        return v

    def create(self, relation):
        """ Get the similarity between two entities of relation
        """
        assert(isinstance(relation, Relation))

        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)
        v1 = self.__get_mean_word2vec_vector(self.stemmer.lemmatize_to_list(e1.value))
        v2 = self.__get_mean_word2vec_vector(self.stemmer.lemmatize_to_list(e2.value))
        return np.array([sum(map(lambda x, y: x * y, v1, v2))])

    def feature_names(self):
        return [self.__class__.__name__]
