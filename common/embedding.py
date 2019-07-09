import numpy as np
from gensim.models.word2vec import Word2Vec


class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, matrix, words):
        assert(isinstance(matrix, np.ndarray) and len(matrix.shape) == 2)
        assert(isinstance(words, list) and len(words) == matrix.shape[0])
        self.__matrix = matrix
        self.__words = words
        self.__index_by_word = self.__create_index(words)

    @property
    def VectorSize(self):
        return self.__matrix.shape[1]

    @property
    def VocabularySize(self):
        return self.__matrix.shape[0]

    @classmethod
    def from_word2vec_format(cls, filepath, binary):
        assert(isinstance(binary, bool))

        w2v_model = Word2Vec.load_word2vec_format(filepath, binary=binary)
        words_count = len(w2v_model.wv.vocab)

        return cls(matrix=np.array([vector for vector in w2v_model.syn0]),
                   words=[w2v_model.wv.index2word[index] for index in range(words_count)])

    @classmethod
    def from_list_with_embedding_func(cls, words, embedding_func):
        assert(isinstance(words, list))
        assert(callable(embedding_func))

        matrix = []
        for word in words:
            vector = embedding_func(word)
            matrix.append(vector)

        return cls(matrix=np.array(matrix),
                   words=words)

    def __create_index(self, words):
        index = {}
        for i, word in enumerate(words):
            index[word] = i
        return index

    def iter_vocabulary(self):
        for word in self.__words:
            yield word, self.__index_by_word[word]

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self.__matrix[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        return self.__words[index]

    def find_index_by_word(self, word):
        assert(isinstance(word, unicode))
        return self.__index_by_word[word]

    def __contains__(self, word):
        assert(isinstance(word, unicode))
        return word in self.__index_by_word

    def __getitem__(self, word):
        assert(isinstance(word, unicode))
        index = self.__index_by_word[word]
        return self.__matrix[index]

