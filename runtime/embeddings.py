from gensim.models.word2vec import Word2Vec
from ..env import stemmer


class Embedding:
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, w2v_model):
        assert(isinstance(w2v_model, Word2Vec))
        self.w2v_model = w2v_model

    def word2index(self, word):
        assert(type(word) == unicode)
        # TODO: Implement word2index.
        return

    def index2word(self, index):
        assert(type(index) == int)
        # TODO: Implement index2word.
        pass

    def __contains__(self, item):
        assert(type(item) == unicode)
        return item in self.w2v_model

    def __getitem__(self, item):
        assert(type(item) == unicode)
        return self.w2v_model[item]


class RusvectoresEmbedding(Embedding):

    def __init__(self, w2v_model):
        assert(isinstance(w2v_model, Word2Vec))
        self.w2v_model = w2v_model

    def __contains__(self, term):
        assert(type(term) == unicode)
        item = self._lemmatize_term_to_rusvectores(term)
        return item in self.w2v_model

    def __getitem__(self, item):
        assert(type(item) == unicode)
        return self.w2v_model[self._lemmatize_term_to_rusvectores(item)]

    @staticmethod
    def _lemmatize_term_to_rusvectores(term):
        """ combine lemmatized 'text' with POS tag (part of speech). """
        assert(type(term) == unicode)
        return stemmer.lemmatize_to_str(term) + '_' + stemmer.get_term_pos(term)
