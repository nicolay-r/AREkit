from gensim.models.word2vec import Word2Vec
from ..env import stemmer


class Embedding(object):
    """
    Represents default wrapper over W2V API.
    """

    def __init__(self, w2v_model):
        assert(isinstance(w2v_model, Word2Vec))
        self.w2v_model = w2v_model

    @staticmethod
    def from_word2vec_filepath(filepath, is_binary):
        w2v_model = Word2Vec.load_word2vec_format(filepath, binary=is_binary)
        return Embedding(w2v_model)

    def word2index(self, word):
        assert(type(word) == unicode)
        # TODO: Implement word2index.
        return

    def index2word(self, index):
        assert(type(index) == int)
        # TODO: Implement index2word.
        pass

    def similarity(self, word_1, word_2):
        assert(type(word_1) == unicode)
        assert(type(word_2) == unicode)
        # TODO: Implement similarity.
        pass

    def __contains__(self, item):
        assert(type(item) == unicode)
        print item
        return item in self.w2v_model

    def __getitem__(self, item):
        assert(type(item) == unicode)
        print item
        return self.w2v_model[item]


class RusvectoresEmbedding(Embedding):

    def __init__(self, w2v_model):
        assert(isinstance(w2v_model, Word2Vec))
        super(RusvectoresEmbedding, self).__init__(w2v_model)

    @staticmethod
    def from_word2vec_filepath(filepath, is_binary):
        w2v_model = Word2Vec.load_word2vec_format(filepath, binary=is_binary)
        return RusvectoresEmbedding(w2v_model)

    def __contains__(self, term):
        assert(type(term) == unicode)
        item = self._lemmatize_term_to_rusvectores(term)
        super(RusvectoresEmbedding, self).__contains__(item)

    def __getitem__(self, term):
        assert(type(term) == unicode)
        item = self._lemmatize_term_to_rusvectores(term)
        super(RusvectoresEmbedding, self).__getitem__(item)

    @staticmethod
    def _lemmatize_term_to_rusvectores(term):
        """ combine lemmatized 'text' with POS tag (part of speech). """
        assert(type(term) == unicode)
        return '_'.join([stemmer.lemmatize_to_str(term), stemmer.get_term_pos(term)])
