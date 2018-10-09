from gensim.models.word2vec import Word2Vec
from core.processing.lemmatization.base import Stemmer


class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, w2v_model, stemmer):
        assert(isinstance(w2v_model, Word2Vec))
        assert(isinstance(stemmer, Stemmer))
        self.w2v_model = w2v_model
        self.stemmer = stemmer

    @property
    def vector_size(self):
        return self.w2v_model.vector_size

    @property
    def vocab(self):
        return self.w2v_model.wv.vocab

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self.w2v_model.syn0[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        return self.w2v_model.wv.index2word[index]

    def find_index_by_word(self, word):
        assert(isinstance(word, unicode))
        return self.w2v_model.wv.index2word.index(word)

    def similarity(self, word_1, word_2):
        assert(isinstance(word_1, unicode))
        assert(isinstance(word_2, unicode))
        # TODO: Implement similarity.
        pass

    def __contains__(self, item):
        assert(isinstance(item, unicode))
        return item in self.w2v_model

    def __getitem__(self, item):
        assert(isinstance(item, unicode))
        return self.w2v_model[item]

