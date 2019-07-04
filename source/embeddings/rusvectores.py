import numpy as np

from core.processing.lemmatization.base import Stemmer
from core.source.embeddings.base import Embedding


class RusvectoresEmbedding(Embedding):

    def __init__(self, matrix, words):
        super(RusvectoresEmbedding, self).__init__(matrix=matrix,
                                                   words=words)

        self.__index_without_pos = self.__create_terms_without_pos()
        self.__stemmer = None

    def set_stemmer(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer

    def create_term_embedding(self, word, max_part_size=3):
        assert(isinstance(word, unicode))

        if word in self:
            return self[word]

        c_i = 0
        c_l = max_part_size
        count = 0
        result_v = np.zeros(self.VectorSize)

        while c_i < len(word):

            if c_l == 0:
                c_i += 1
                c_l = max_part_size
                continue

            right_b = min(len(word), c_i+c_l)
            s_i = self.__find_index(term=word[c_i:right_b],
                                    lemmatize=False)

            if s_i is None:
                c_l -= 1
                continue

            result_v += self.get_vector_by_index(s_i)
            c_i += c_l
            count += 1

        return result_v / count if count > 0 else result_v

    def find_index_by_word(self, word):
        assert(isinstance(word, unicode))
        return self.__find_index(word)

    def __find_index(self, term, lemmatize=True):
        assert(isinstance(term, unicode))
        assert(isinstance(lemmatize, bool))
        if lemmatize:
            term = self.__stemmer.lemmatize_to_str(term)
        return self.__index_without_pos[term] \
            if term in self.__index_without_pos else None

    def __create_terms_without_pos(self):
        d = {}
        for word_with_pos, index in self.iter_vocabulary():
            word = word_with_pos.split('_')[0]
            if word in d:
                continue
            d[word] = index

        return d

    def __contains__(self, term):
        assert(isinstance(term, unicode))
        return self.__find_index(term) is not None

    def __getitem__(self, term):
        assert(isinstance(term, unicode))
        index = self.__find_index(term)
        return self.get_vector_by_index(index)
