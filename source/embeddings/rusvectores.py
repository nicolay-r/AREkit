from core.source.embeddings.base import Embedding
from core.processing.lemmatization.base import Stemmer
from gensim.models.word2vec import Word2Vec


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
        assert(isinstance(term, unicode))
        assert(isinstance(stemmer, Stemmer))

        term = stemmer.lemmatize_to_str(term)
        pos = stemmer.get_term_pos(term)
        if pos is stemmer.get_pos_uknown_token():
            return None
        return '_'.join([term, pos])


