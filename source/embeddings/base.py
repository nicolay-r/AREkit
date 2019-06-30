import numpy as np
from gensim.models.word2vec import Word2Vec
from core.processing.lemmatization.base import Stemmer
from core.processing.pos.base import POSTagger


# TODO. Remove pos tagger.
class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, matrix, words, stemmer=None, pos_tagger=None):
        assert(isinstance(matrix, np.ndarray) and len(matrix.shape) == 2)
        assert(isinstance(words, list) and len(words) == matrix.shape[0])
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        self.__matrix = matrix
        self.__words = words
        self.__stemmer = stemmer
        self.__pos_tagger = pos_tagger
        self.__index_by_word = self.__create_index(words)

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def VectorSize(self):
        return self.__matrix.shape[1]

    @property
    def VocabularySize(self):
        return self.__matrix.shape[0]

    @classmethod
    def from_word2vec_format(cls, filepath, binary, stemmer=None, pos_tagger=None):
        assert(isinstance(binary, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        assert(isinstance(pos_tagger, POSTagger) or pos_tagger is None)

        w2v_model = Word2Vec.load_word2vec_format(filepath, binary=binary)
        words_count = len(w2v_model.wv.vocab)

        return cls(matrix=np.array([vector for vector in w2v_model.syn0]),
                   words=[w2v_model.wv.index2word[index] for index in range(words_count)],
                   stemmer=stemmer,
                   pos_tagger=pos_tagger)

    def __create_index(self, words):
        index = {}
        for i, word in enumerate(words):
            index[word] = i
        return index

    @property
    def iter_vocabulary(self):
        for word, index in self.__words:
            yield word, self.__index_by_word[word]

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self.__matrix[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        return self.__words[index]

    def find_index_by_word(self, word, return_unknown=False):
        assert(isinstance(word, unicode))
        assert(return_unknown is False)
        return self.__index_by_word[word]

    def __contains__(self, word):
        assert(isinstance(word, unicode))
        return word in self.__index_by_word

    def __getitem__(self, word):
        assert(isinstance(word, unicode))
        index = self.__index_by_word[word]
        return self.__matrix[index]

