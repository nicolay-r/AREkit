from arekit.common.text.stemmer import Stemmer
from arekit.contrib.networks.embedding import Embedding


class RusvectoresEmbedding(Embedding):
    """ Wrapper over models from the following resource.
        https://rusvectores.org/ru/models/

        NOTE: Usually these are embeddings for texts written in Russian.
        for the better performance it is expected that we adopt stemmer.
    """

    def __init__(self, matrix, words, stemmer):
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        super(RusvectoresEmbedding, self).__init__(matrix=matrix, words=words)
        self.__index_without_pos = self.__create_terms_without_pos()
        self.__stemmer = stemmer
        self.__lemmatize_by_default = stemmer is not None

    def try_find_index_by_plain_word(self, word):
        assert(isinstance(word, str))

        temp = self.__lemmatize_by_default
        self.__lemmatize_by_default = False
        index = super(RusvectoresEmbedding, self).try_find_index_by_plain_word(word)
        self.__lemmatize_by_default = temp

        return index

    def _handler(self, word):
        return self.__try_find_word_index_pair_lemmatized(word, self.__lemmatize_by_default)

    # region private methods

    def __try_find_word_index_pair_lemmatized(self, term, lemmatize):
        assert(isinstance(term, str))
        assert(isinstance(lemmatize, bool))

        if lemmatize:
            term = self.__stemmer.lemmatize_to_str(term)

        index = self.__index_without_pos[term] \
            if term in self.__index_without_pos else None

        return term, index

    def __create_terms_without_pos(self):
        d = {}
        for word_with_pos, index in self.iter_vocabulary():
            assert(isinstance(word_with_pos, str))
            word = word_with_pos.split(u'_')[0]
            if word in d:
                continue
            d[word] = index

        return d

    # endregion