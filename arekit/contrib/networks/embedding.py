from collections.abc import Iterable
import numpy as np


class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, matrix, words):
        assert(isinstance(matrix, np.ndarray) and len(matrix.shape) == 2)
        assert(isinstance(words, np.ndarray))
        assert(len(words) == matrix.shape[0])
        self._matrix = matrix
        self.__words = words
        self.__index_by_word = self.__create_index(words)

    # region properties

    @property
    def VectorSize(self):
        return self._matrix.shape[1]

    @property
    def VocabularySize(self):
        return self._matrix.shape[0]

    # endregion

    # region classmethods

    @classmethod
    def from_word_embedding_pairs_iter(cls, word_embedding_pairs):
        assert(isinstance(word_embedding_pairs, Iterable))

        matrix = []
        words = []
        used = set()
        for word, vector in word_embedding_pairs:

            if word in used:
                continue

            used.add(word)

            matrix.append(vector)
            words.append(word)

        return cls(matrix=np.array(matrix) if len(matrix) > 0 else np.empty(shape=(0, 0)),
                   words=np.array(words))

    @classmethod
    def from_list_with_embedding_func(cls, words_iter, embedding_func):
        assert(isinstance(words_iter, Iterable))
        assert(callable(embedding_func))

        matrix = []
        words = []
        used = set()
        for word in words_iter:

            if word in used:
                continue
            used.add(word)

            vector = embedding_func(word)
            matrix.append(vector)
            words.append(word)

        return cls(matrix=np.array(matrix),
                   words=words)

    # endregion

    # region private methods

    @staticmethod
    def __create_index(words):
        index = {}
        for i, word in enumerate(words):
            index[word] = i
        return index

    def __try_find_word_index_pair(self, word):
        """
        Assumes to pefrom term transformation (optional)
        in order to find a term in an inner vocabulary

        returns: pair
            (processed_term, index)
        """
        assert(isinstance(word, str))

        has_index = self.__index_by_word[word] if word in self.__index_by_word else None
        word = word if has_index else None
        return word, has_index

    def __hadler_core(self, word):
        """
        Core word handler.
        Assumes to perform word stripping.
        """
        stripped_word = word.strip()
        return self._handler(stripped_word)

    # endregion

    def iter_vocabulary(self):
        for word in self.__words:
            yield word, self.__index_by_word[word]

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self._matrix[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        return self.__words[index]

    def try_find_index_by_word(self, word):
        assert(isinstance(word, str))
        _, index = self.__hadler_core(word)
        return index

    def try_find_index_by_plain_word(self, word):
        assert(isinstance(word, str))
        _, index = self.__hadler_core(word)
        return index

    def try_get_related_word(self, word):
        word, _ = self.__hadler_core(word)
        return word

    def _handler(self, word):
        return self.__try_find_word_index_pair(word)

    # region overriden methods

    def __contains__(self, word):
        assert(isinstance(word, str))
        _, index = self.__hadler_core(word)
        return index is not None

    def __getitem__(self, word):
        assert(isinstance(word, str))
        _, index = self.__hadler_core(word)
        return self._matrix[index]

    # endregion
