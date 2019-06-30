from core.source.embeddings.base import Embedding
from gensim.models.word2vec import Word2Vec


class RusvectoresEmbedding(Embedding):

    def __init__(self, matrix, words, stemmer, pos_tagger):
        super(RusvectoresEmbedding, self).__init__(matrix=matrix,
                                                   words=words,
                                                   stemmer=stemmer,
                                                   pos_tagger=pos_tagger)

    # TODO. Provide API for complex embedding generation.
    def create_term_embedding(self, word):
        assert(isinstance(word, unicode))
        assert(word not in self)
        pass

    def find_index_by_word(self, word, return_unknown=False):
        assert(isinstance(word, unicode))
        assert(return_unknown is False)

        item = self.__lemmatize_term_to_rusvectores(word)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).find_index_by_word(item)

    def __lemmatize_term_to_rusvectores(self, term):
        """
        Combine lemmatized 'text' with POS tag (part of speech).
        """
        assert(isinstance(term, unicode))

        term = self.Stemmer.lemmatize_to_str(term)
        pos = self.PosTagger.get_term_pos(term)
        if pos is self.PosTagger.Unknown:
            return None
        return '_'.join([term, pos])

    def __contains__(self, term):
        assert(isinstance(term, unicode))

        item = self.__lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__contains__(item)

    def __getitem__(self, term):
        assert(isinstance(term, unicode))

        item = self.__lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__getitem__(item)
