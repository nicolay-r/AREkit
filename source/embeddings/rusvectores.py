import logging
import numpy as np
from arekit.processing.lemmatization.base import Stemmer
from arekit.common.embeddings.base import Embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RusvectoresEmbedding(Embedding):

    def __init__(self, matrix, words):
        super(RusvectoresEmbedding, self).__init__(matrix=matrix,
                                                   words=words)

        self.__index_without_pos = self.__create_terms_without_pos()
        self.__stemmer = None

    def set_stemmer(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer

    def create_term_embedding(self, term, max_part_size=3):
        assert(isinstance(term, unicode))

        count = 0
        vector = np.zeros(self.VectorSize)
        for word in term.split(u' '):
            v, c = self.__create_embedding_for_word(word=word,
                                                    max_part_size=max_part_size)
            count += c
            vector += v

        return vector / count if count > 0 else vector

    def __create_embedding_for_word(self, word, max_part_size):
        assert(isinstance(word, unicode))

        if word in self:
            return self[word], 1

        c_i = 0
        c_l = max_part_size
        count = 0
        vector = np.zeros(self.VectorSize)
        missings = []

        while c_i < len(word):

            if c_l == 0:
                missings.append(c_i)
                c_i += 1
                c_l = max_part_size
                continue

            right_b = min(len(word), c_i + c_l)
            s_i = self.__try_find_index(term=word[c_i:right_b],
                                        lemmatize=False)

            if s_i is None:
                c_l -= 1
                continue
            vector += self.get_vector_by_index(s_i)
            c_i += c_l
            count += 1

        w_debug = u''.join([u'?' if i in missings else ch
                            for i, ch in enumerate(word)])
        logger.debug(u'Embedded: {}'.format(w_debug).encode('utf-8'))

        return vector, count

    def try_find_index_by_word(self, word, lemmatize=True):
        assert(isinstance(word, unicode))
        return self.__try_find_index(term=word,
                                     lemmatize=lemmatize)

    def __try_find_index(self, term, lemmatize=True):
        assert(isinstance(term, unicode))
        assert(isinstance(lemmatize, bool))

        if lemmatize:
            term = self.__stemmer.lemmatize_to_str(term)

        return self.__index_without_pos[term] \
            if term in self.__index_without_pos else None

    def __create_terms_without_pos(self):
        d = {}
        for word_with_pos, index in self.iter_vocabulary():
            assert(isinstance(word_with_pos, unicode))
            word = word_with_pos.split(u'_')[0]
            if word in d:
                continue
            d[word] = index

        return d

    def __contains__(self, term):
        assert(isinstance(term, unicode))
        return self.__try_find_index(term) is not None

    def __getitem__(self, term):
        assert(isinstance(term, unicode))
        index = self.__try_find_index(term)
        return self.get_vector_by_index(index)
