from gensim.models.word2vec import Word2Vec
from core.processing.lemmatization.base import Stemmer
from core.source.tokens import Tokens
import utils


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
        assert(type(index) == int)
        return self.w2v_model.syn0[index]

    def get_word_by_index(self, index):
        assert(type(index) == int)
        return self.w2v_model.wv.index2word[index]

    def find_index_by_word(self, word):
        assert(type(word) == unicode)
        return self.w2v_model.wv.index2word.index(word)

    def similarity(self, word_1, word_2):
        assert(type(word_1) == unicode)
        assert(type(word_2) == unicode)
        # TODO: Implement similarity.
        pass

    def __contains__(self, item):
        assert(type(item) == unicode)
        return item in self.w2v_model

    def __getitem__(self, item):
        assert(type(item) == unicode)
        return self.w2v_model[item]


class RusvectoresEmbedding(Embedding):

    def __init__(self, w2v_model, stemmer):
        assert(isinstance(w2v_model, Word2Vec))
        assert(isinstance(stemmer, Stemmer))
        super(RusvectoresEmbedding, self).__init__(w2v_model, stemmer)

    def __contains__(self, term):
        assert(type(term) == unicode)

        item = self._lemmatize_term_to_rusvectores(term, self.stemmer)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__contains__(item)

    def __getitem__(self, term):
        assert(type(term) == unicode)

        item = self._lemmatize_term_to_rusvectores(term, self.stemmer)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__getitem__(item)

    def find_index_by_word(self, word):
        assert(type(word) == unicode)

        item = self._lemmatize_term_to_rusvectores(word, self.stemmer)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).find_index_by_word(item)

    @staticmethod
    def _lemmatize_term_to_rusvectores(term, stemmer):
        """
        Combine lemmatized 'text' with POS tag (part of speech).
        """
        assert(type(term) == unicode)
        assert(isinstance(stemmer, Stemmer))

        term = stemmer.lemmatize_to_str(term)
        pos = stemmer.get_term_pos(term)
        if pos is stemmer.get_pos_unknown_token():
            return None
        return '_'.join([term, pos])


class TokenEmbeddingVectors:
    """
    Embedding vectors for text punctuation, based on Tokens
    """

    tokens = [Tokens.COMMA,
              Tokens.COLON,
              Tokens.SEMICOLON,
              Tokens.QUOTE,
              Tokens.DASH,
              Tokens.EXC_SIGN,
              Tokens.QUESTION_SIGN,
              Tokens.OPEN_BRACKET,
              Tokens.CLOSED_BRACKET,
              Tokens.LONG_DASH,
              Tokens.NUMBER,
              Tokens.TRIPLE_DOTS,
              Tokens.UNKNOWN_CHAR,
              Tokens.UNKNOWN_WORD]

    def __init__(self, vector_size):
        self.E = {}
        for index, token in enumerate(self.tokens):
            self.E[token] = utils.get_random_vector(vector_size, index)

    @staticmethod
    def count():
        return len(TokenEmbeddingVectors.tokens)

    @staticmethod
    def get_token_index(token):
        assert(isinstance(token, unicode))
        return TokenEmbeddingVectors.tokens.index(token)

    def __getitem__(self, token):
        assert (isinstance(token, unicode))
        return self.E[token]

    def __iter__(self):
        for token in self.E:
            yield token

    def __contains__(self, token):
        assert (isinstance(token, unicode))
        return token in self.E


