import numpy as np

from arekit.contrib.networks.vectorizer import BaseVectorizer


class RandomNormalVectorizer(BaseVectorizer):

    def __init__(self, vector_size, token_offset=12345):
        assert(isinstance(vector_size, int))
        self.__vector_size = vector_size
        self.__seed_token_offset = token_offset

    def create_term_embedding(self, term):
        """ term: is its index.
        """
        assert(isinstance(term, int))

        return self.__get_random_normal_distribution(
            vector_size=self.__vector_size,
            seed=term + self.__seed_token_offset,
            loc=0.05,
            scale=0.025)

    # region private methods

    @staticmethod
    def __get_random_normal_distribution(vector_size, seed, loc, scale):
        assert (isinstance(vector_size, int))
        assert (isinstance(seed, int))
        np.random.seed(seed)
        return np.random.normal(loc=loc, scale=scale, size=vector_size)

    # endregion
