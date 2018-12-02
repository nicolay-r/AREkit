from core.source.embeddings.base import Embedding
from core.processing.pos.base import POSTagger
from core.processing.lemmatization.base import Stemmer
from gensim.models.word2vec import Word2Vec


class RusvectoresEmbedding(Embedding):

    def __init__(self, w2v_model, stemmer, pos_tagger):
        assert(isinstance(w2v_model, Word2Vec))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(pos_tagger, POSTagger))
        super(RusvectoresEmbedding, self).__init__(w2v_model)
        self.stemmer = stemmer
        self.pos_tagger = pos_tagger

    def __contains__(self, term):
        assert(type(term) == unicode)

        item = self._lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__contains__(item)

    def __getitem__(self, term):
        assert(type(term) == unicode)

        item = self._lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__getitem__(item)

    def find_index_by_word(self, word):
        assert(type(word) == unicode)

        item = self._lemmatize_term_to_rusvectores(word)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).find_index_by_word(item)

    def _lemmatize_term_to_rusvectores(self, term):
        """
        Combine lemmatized 'text' with POS tag (part of speech).
        """
        assert(isinstance(term, unicode))

        term = self.stemmer.lemmatize_to_str(term)
        pos = self.pos_tagger.get_term_pos(term)
        if pos is self.pos_tagger.get_pos_unknown_token():
            return None
        return '_'.join([term, pos])


