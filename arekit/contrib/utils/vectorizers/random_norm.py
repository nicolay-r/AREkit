import numpy as np

from arekit.contrib.networks.vectorizer import BaseVectorizer


class RandomNormalVectorizer(BaseVectorizer):

    def __init__(self, term_to_index_func, vector_size, token_offset=12345):
        assert(isinstance(vector_size, int))
        assert(callable(term_to_index_func))
        self.__vector_size = vector_size
        self.__seed_token_offset = token_offset
        self.__term_to_index_func = term_to_index_func

    def create_term_embedding(self, term):
        """ term: is its index.
        """
        embedding = self.__get_random_normal_distribution(
            vector_size=self.__vector_size,
            seed=self.__term_to_index_func(term) + self.__seed_token_offset,
            loc=0.05,
            scale=0.025)
        return term, embedding

    # region private methods

    @staticmethod
    def __get_random_normal_distribution(vector_size, seed, loc, scale):
        assert (isinstance(vector_size, int))
        assert (isinstance(seed, int))
        np.random.seed(seed)
        return np.random.normal(loc=loc, scale=scale, size=vector_size)

    # endregion
