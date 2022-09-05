import numpy as np

from arekit.contrib.networks.vectorizer import BaseVectorizer


class RandomNormalVectorizer(BaseVectorizer):

    def __init__(self, vector_size, token_offset=12345, max_tokens_count=100):
        assert(isinstance(vector_size, int))
        self.__vector_size = vector_size
        self.__seed_token_offset = token_offset
        self.__max_tokens_count = max_tokens_count

    def create_term_embedding(self, term):
        """ term: is its index.
        """
        embedding = self.__get_random_normal_distribution(
            vector_size=self.__vector_size,
            seed=(self.__string_to_int(term) % self.__max_tokens_count) + self.__seed_token_offset,
            loc=0.05,
            scale=0.025)
        return term, embedding

    # region private methods

    def __string_to_int(self, s):
        # Originally taken from here:
        # https://stackoverflow.com/questions/2511058/persistent-hashing-of-strings-in-python
        ord3 = lambda x: '%.3d' % ord(x)
        return int(''.join(map(ord3, s)))

    @staticmethod
    def __get_random_normal_distribution(vector_size, seed, loc, scale):
        assert (isinstance(vector_size, int))
        assert (isinstance(seed, int))
        np.random.seed(seed)
        return np.random.normal(loc=loc, scale=scale, size=vector_size)

    # endregion
