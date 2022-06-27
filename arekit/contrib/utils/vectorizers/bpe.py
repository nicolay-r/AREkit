import numpy as np

from arekit.common.log_utils import logger
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.networks.vectorizer import BaseVectorizer


class BPEVectorizer(BaseVectorizer):
    """ Embedding algorithm based on parts (trigrams originally)
    """

    def __init__(self, embedding, max_part_size=3, word_separator=' '):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(max_part_size, int))
        self.__embedding = embedding
        self.__max_part_size = max_part_size
        self.__word_separator = word_separator

    def create_term_embedding(self, term):
        """ Note: returns modified term value in a form of the `word` returning parameter.
        """
        assert(isinstance(term, str))

        word, word_embedding = self.__get_from_embedding(term=term) \
            if term in self.__embedding else self.__compose_from_parts(term=term)

        # In order to prevent a problem of the further separations during reading process.
        # it is necessary to replace the separators with the other chars.
        word = word.replace(self.__word_separator, '-')

        return word, word_embedding

    def __compose_from_parts(self, term, do_lowercase=True):
        # remove empty spaces before and after.
        term = term.strip()

        # perform lowercasing
        if do_lowercase:
            term = term.lower()

        # Calculating vector from term parts
        count = 0
        vector = np.zeros(self.__embedding.VectorSize)
        for word in term.split(self.__word_separator):
            v, c = self.__create_embedding_for_word(word=word, embedding=self.__embedding)
            count += c
            vector = vector + v

        return term, vector / count if count > 0 else vector

    def __get_from_embedding(self, term):
        return self.__embedding.try_get_related_word(term), self.__embedding[term]

    def __create_embedding_for_word(self, word, embedding):
        assert(isinstance(word, str))
        assert(isinstance(embedding, Embedding))

        if word in embedding:
            return embedding[word], 1

        c_i = 0
        c_l = self.__max_part_size
        count = 0
        vector = np.zeros(embedding.VectorSize)
        missings = []

        while c_i < len(word):

            if c_l == 0:
                missings.append(c_i)
                c_i += 1
                c_l = self.__max_part_size
                continue

            right_b = min(len(word), c_i + c_l)

            s_i = embedding.try_find_index_by_plain_word(word=word[c_i:right_b])

            if s_i is None:
                c_l -= 1
                continue

            vector += embedding.get_vector_by_index(s_i)
            c_i += c_l
            count += 1

        debug = False
        if debug:
            w_debug = ''.join(['?' if i in missings else ch
                               for i, ch in enumerate(word)])
            logger.debug('Embedded: {}'.format(w_debug).encode('utf-8'))

        return vector, count
