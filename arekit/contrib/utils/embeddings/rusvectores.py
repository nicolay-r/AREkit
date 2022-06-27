import numpy as np
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.networks.embeddings.base import Embedding
from gensim.models import KeyedVectors


class RusvectoresEmbedding(Embedding):

    def __init__(self, matrix, words):
        super(RusvectoresEmbedding, self).__init__(matrix=matrix,
                                                   words=words)

        self.__index_without_pos = self.__create_terms_without_pos()
        self.__stemmer = None
        self.__lemmatize_by_default = True

    @classmethod
    def from_word2vec_format(cls, filepath, binary):
        assert(isinstance(binary, bool))

        w2v_model = KeyedVectors.load_word2vec_format(filepath, binary=binary)
        words_count = len(w2v_model.wv.vocab)

        return cls(matrix=np.array([vector for vector in w2v_model.syn0]),
                   words=[w2v_model.wv.index2word[index] for index in range(words_count)])

    def set_stemmer(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer

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