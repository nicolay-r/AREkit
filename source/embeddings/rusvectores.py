import logging
from arekit.processing.lemmatization.base import Stemmer
from arekit.common.embeddings.base import Embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RusvectoresEmbedding(Embedding):

    __lemmatize_by_default = True

    def __init__(self, matrix, words):
        super(RusvectoresEmbedding, self).__init__(matrix=matrix,
                                                   words=words)

        self.__index_without_pos = self.__create_terms_without_pos()
        self.__stemmer = None

    def set_stemmer(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer

    def try_find_index_by_word(self, word):
        assert(isinstance(word, unicode))
        _, index =self.__try_find_word_index_pair_lemmatized(
            term=word,
            lemmatize=self.__lemmatize_by_default)
        return index

    def try_find_index_by_plain_word(self, word):
        assert(isinstance(word, unicode))
        _, index = self.__try_find_word_index_pair_lemmatized(
            term=word,
            lemmatize=False)
        return index

    def try_get_related_word(self, word):
        assert(isinstance(word, unicode))
        word, _ = self.__try_find_word_index_pair_lemmatized(
            term=word,
            lemmatize=self.__lemmatize_by_default)
        return word

    # region private methods

    def __try_find_word_index_pair_lemmatized(self, term, lemmatize):
        assert(isinstance(term, unicode))
        assert(isinstance(lemmatize, bool))

        if lemmatize:
            term = self.__stemmer.lemmatize_to_str(term)

        index = self.__index_without_pos[term] \
            if term in self.__index_without_pos else None

        return term, index

    def __create_terms_without_pos(self):
        d = {}
        for word_with_pos, index in self.iter_vocabulary():
            assert(isinstance(word_with_pos, unicode))
            word = word_with_pos.split(u'_')[0]
            if word in d:
                continue
            d[word] = index

        return d

    # endregion

    # region general methods

    def __contains__(self, term):
        assert(isinstance(term, unicode))
        _, index = self.__try_find_word_index_pair_lemmatized(term, self.__lemmatize_by_default)
        return index is not None

    def __getitem__(self, term):
        assert(isinstance(term, unicode))
        _, index = self.__try_find_word_index_pair_lemmatized(term, self.__lemmatize_by_default)
        return self._matrix[index]

    # endregion
