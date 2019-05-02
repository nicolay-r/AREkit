from gensim.models.word2vec import Word2Vec
from core.processing.lemmatization.base import Stemmer
from core.processing.pos.base import POSTagger


class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, w2v_model=None, stemmer=None, pos_tagger=None):
        assert(isinstance(w2v_model, Word2Vec) or w2v_model is None)
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        assert(isinstance(pos_tagger, POSTagger) or pos_tagger is None)
        self.w2v_model = w2v_model
        self._stemmer = stemmer
        self._pos_tagger = pos_tagger

    @classmethod
    def from_file(cls, filepath, binary, stemmer=None, pos_tagger=None):
        assert(isinstance(binary, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        assert(isinstance(pos_tagger, POSTagger) or pos_tagger is None)

        w2v_model = Word2Vec.load_word2vec_format(filepath, binary=binary)

        return cls(w2v_model, stemmer, pos_tagger)

    @property
    def Stemmer(self):
        return self._stemmer

    @property
    def PosTagger(self):
        return self._pos_tagger

    @property
    def VectorSize(self):
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

    def find_index_by_word(self, word, return_unknown=False):
        assert(isinstance(word, unicode))
        assert(return_unknown is False)
        return self.w2v_model.wv.index2word.index(word)

    def similarity(self, word_1, word_2):
        assert(isinstance(word_1, unicode))
        assert(isinstance(word_2, unicode))
        # TODO: Implement similarity.
        pass

    def __contains__(self, word):
        assert(isinstance(word, unicode))
        return word in self.w2v_model

    def __getitem__(self, word):
        assert(isinstance(word, unicode))
        return self.w2v_model[word]

